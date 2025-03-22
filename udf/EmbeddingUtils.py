import asyncio
import json
from collections import defaultdict
from typing import NamedTuple
from tenacity import retry, stop_after_attempt, wait_incrementing

import aiohttp
import shelve
import tqdm
import numpy as np
import pandas as pd

from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from pymilvus.orm import utility
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class LLMDao:
    def __init__(self):
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.key = "82caa1a3bc8e40a9af073d8727db75f1.Vx8ouQlTK5fMmbN2"
        self.model = "GLM-4-Plus"


    @retry(stop=stop_after_attempt(6), wait=wait_incrementing(start=1, increment=1))
    async def _fetch(self, session, prompt: str, semaphore: asyncio.Semaphore, pbar: tqdm.tqdm) -> str:
        async with semaphore:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}"
            }
            request_body = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "do_sample": False,
            }
            async with session.post(self.url, headers=headers, json=request_body) as response:
                response.raise_for_status()
                response_data = await response.json()
                choices = response_data.get("choices", [])
                if not choices:
                    return "Error: Missing or empty 'choices' in response"
                pbar.update(1)
                return choices[0].get("message", {}).get("content", "")

    async def _bulk_fetch(self, prompts: list[str], max_concurrent_requests) -> list[str]:
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        async with aiohttp.ClientSession() as session:
            with tqdm.tqdm(total=len(prompts), desc="Requesting to LLM") as pbar:
                tasks = [self._fetch(session, prompt, semaphore, pbar) for prompt in prompts]
            return await asyncio.gather(*tasks)

    def _bulk_get_response(self, prompts: list[str], max_concurrent_requests: int = 20) -> list[str]:
        return asyncio.run(self._bulk_fetch(prompts, max_concurrent_requests))

    def provide_keywords(self, decription: str):
        prompt = f"""
        Below is a description intended for searching specific information. 
        Please provide a set of relevant keywords from the description to improve the search results, 
        separated by commas and ranked by relevance. Ensure to output only the keywords without any additional text.
        Description: {decription}
        """
        response = self._bulk_get_response([prompt])[0]
        return response

    def describe_relation(self, pairs: list[tuple[str, str]]) -> list[str]:
        prompts = [
            f"""
            You are a concept master. 
            Please provide a concise and specific description of the relationship 
            between "{pair[0]}" and "{pair[1]}" in one sentence using no more than 10 words and without any punctuation. 
            Only the summary result is needed. Ensure the summary does not exceed 10 words and is concise and logical phrase.
            """
            for pair in pairs
        ]
        return self._bulk_get_response(prompts)

    def summarize(self, path_clusters: list[list[str]]) -> list[str]:
        prompts = [
            """
            You are a summarization expert. 
            I will provide several lines of phrases delimited by commas.
            Please summarize them into one sentence using no more than 10 words characters and without any punctuation. 
            Only the summary result is needed. Ensure the summary does not exceed 10 words and is concise and logical phrase.
            
            Here are the phrases to summarize:
            {}
            """.format("\n".join(path_cluster))
            for path_cluster in path_clusters
        ]
        return self._bulk_get_response(prompts)


if __name__ == '__main__':
    # 本地测试
    keywords = LLMDao().summarize([
        ['customer.c_acctbal', 'orders.o_comment', 'supplier.s_acctbal'],
        ['customer.c_address', 'lineitem.l_returnflag'],
        ['customer.c_comment', 'customer.c_mktsegment', 'nation.n_regionkey'],
        ['customer.c_custkey', 'customer.c_name', 'nation.n_comment', 'nation.n_nationkey', 'orders.o_orderstatus', 'supplier.s_suppkey'],
        ['customer.c_nationkey', 'orders.o_clerk', 'orders.o_shippriority'],
        ['customer.c_phone', 'lineitem.l_commitdate', 'region.r_comment', 'region.r_regionkey'],
        ['lineitem.l_comment', 'nation.n_name', 'region.r_name', 'supplier.s_comment', 'supplier.s_nationkey'],
        ['lineitem.l_discount', 'orders.o_orderkey', 'part.p_comment', 'part.p_type'],
        ['lineitem.l_extendedprice', 'lineitem.l_linestatus', 'lineitem.l_suppkey'],
        ['lineitem.l_linenumber', 'orders.o_orderdate', 'orders.o_orderpriority', 'supplier.s_phone'],
        ['lineitem.l_orderkey', 'lineitem.l_quantity'],
        ['lineitem.l_partkey', 'orders.o_custkey', 'orders.o_totalprice'],
        ['lineitem.l_receiptdate', 'lineitem.l_tax'],
        ['lineitem.l_shipdate', 'lineitem.l_shipmode', 'supplier.s_address'],
        ['lineitem.l_shipinstruct', 'part.p_retailprice', 'partsupp.ps_availqty', 'partsupp.ps_suppkey'],
        ['part.p_brand', 'part.p_container', 'part.p_name', 'part.p_size', 'supplier.s_name'],
        ['part.p_mfgr', 'part.p_partkey'],
        ['partsupp.ps_comment', 'partsupp.ps_partkey', 'partsupp.ps_supplycost'],
    ])
    print(keywords)


class Encoder:
    def __init__(self):
        print("Initializing Sentence Transformer")
        self.encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
        print("Initializing Embedding Cache")
        self.embeddings = shelve.open('cache/embeddings', writeback=True)
        self.batch_size = 128
        self.child_weight = 0.3

    def __enter__(self):
        self.embeddings.__enter__()
        return self

    def __exit__(self, type, value, trace):
        print("Closing... EmbeddingCalculator")
        self.embeddings.__exit__(type, value, trace)
        print("Closed")

    def encode(self, descriptions: list[str]) -> list[np.array(np.float32)]:
        print("Calculating embeddings of", len(descriptions), "descriptions")

        uncached_descriptions = [
            description
            for description in tqdm.tqdm(descriptions, desc="Finding uncached descriptions")
            if description not in self.embeddings
        ]
        print("Uncached descriptions:", len(uncached_descriptions))

        description_batches = [
            uncached_descriptions[i:i + self.batch_size]
            for i in range(0, len(uncached_descriptions), self.batch_size)
        ]
        with tqdm.tqdm(total=len(uncached_descriptions), desc="Calculating embeddings") as pbar:
            for description_batch in description_batches:
                embedding_batch = self.encoder.encode(description_batch)
                for description, embedding in zip(description_batch, embedding_batch):
                    self.embeddings[description] = embedding
                self.embeddings.sync()
                pbar.update(len(description_batch))

        return [
            self.embeddings[description]
            for description in tqdm.tqdm(descriptions, desc="Gathering embeddings from cache")
        ]


class MilvusDao:
    def __init__(self, kvargs):
        milvus_host = "localhost"
        milvus_port = 19530
        if kvargs.get("host"):
            milvus_host = kvargs["host"].decode("utf-8")
        if kvargs.get("port"):
            milvus_port = int(kvargs["port"].decode("utf-8"))

        self.batch_size = 1000
        print("Connecting to Milvus")
        connections.connect(host=milvus_host, port=milvus_port)
        # connections.connect(uri=f"cache/milvus.db")
        self.collection = self._create_or_load_collection()

    def _create_or_load_collection(self):
        table_name = 'embeddings'

        if utility.has_collection(table_name):
            print(f"Loading collection {table_name}")
            return Collection(table_name)

        print(f"Creating collection {table_name}")
        fields = [
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4097)
        ]

        schema = CollectionSchema(fields, description="embedding collection")
        collection = Collection(table_name, schema=schema)

        print("Creating index...")
        collection.create_index(field_name="embedding",
                                index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE"})
        collection.create_index(field_name="path", index_params={"index_type": "Trie"})
        return collection

    def delete_all(self):
        print("Recreating collection")
        self.collection.drop()
        self.collection = self._create_or_load_collection()

    def insert_embeddings(self, paths, embeddings, descriptions):
        assert len(paths) == len(embeddings)
        assert len(paths) == len(descriptions)
        insert_results = []
        with tqdm.tqdm(total=len(paths), desc="Inserting embeddings") as pbar:
            for i in range(0, len(paths), self.batch_size):
                paths_batch = paths[i:i + self.batch_size]
                embeddings_batch = embeddings[i:i + self.batch_size]
                descriptions_batch = descriptions[i:i + self.batch_size]
                insert_result = self.collection.insert([paths_batch, embeddings_batch, descriptions_batch])
                insert_results.append(insert_result)
                pbar.update(len(paths_batch))
        return insert_results

    def load(self):
        print("Loading milvus collection")
        self.collection.load()

    def search_similarity(self, embedding, pattern):
        print("Searching similar embeddings")

        expr = None
        if len(pattern) > 0:
            milvus_pattern = pattern.replace("*", "%")
            expr ='path like "' + milvus_pattern +"'"

        entities = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            output_fields=["path"],
            limit=10,
            param={"metric_type": "COSINE"},
            expr=expr,
            search_params = {
                "hints": "iterative_filter"
            }
        )

        return [hit.fields["path"] for hit in entities[0]]


    def fetch_by_path(self, paths_json: str):
        print("fetching by path")
        entities = self.collection.query(
            output_fields=["path", "embedding", "description"],
            expr="path in {paths}",
            expr_params={"paths": json.loads(paths_json)}
        )
        return [(entity["path"], entity["embedding"], entity["description"]) for entity in entities]


    def bulk_search_similarity(self, embedding_list, path_list_json):
        print("Searching similar embeddings")
        entities = self.collection.search(
            expr="path in " + path_list_json,
            data=embedding_list,
            anns_field="embedding",
            output_fields=["path", "description"],
            limit=10,
            param={"metric_type": "COSINE"},
        )

        return [
            [(hit.fields["path"], hit.fields["description"], hit.distance) for hit in entity]
            for entity in entities
        ]


class UDFStoreEmbedding:
    def __init__(self):
        pass

    def _build_descriptions(self, paths: list[str]) -> dict[str, str]:
        print("building descriptions for", len(paths), "paths")
        nodes_list = [path.split(".") for path in paths]
        tree = self._build_path_tree(nodes_list)

        descriptions = {}
        self._dfs_build_descriptions(tree, [], descriptions)
        return descriptions

    def _build_path_tree(self, nodes_list: list[list[str]]) -> dict:
        print("building path tree for", len(nodes_list), "paths")

        tree = {}
        for nodes in nodes_list:
            current_node = tree
            for node in nodes:
                current_node = current_node.setdefault(node, {})
        return tree

    def _dfs_build_descriptions(self, tree, path_nodes: list[str], descriptions: dict[str, str]) -> None:
        for key, value in tree.items():
            path_nodes.append(key)
            description_nodes = list(reversed(path_nodes))
            if not value:
                description_nodes += list(tree.keys())
            else:
                description_nodes += list(value.keys())
            descriptions[".".join(path_nodes)] = ", ".join(description_nodes)
            self._dfs_build_descriptions(value, path_nodes, descriptions)
            path_nodes.pop()

    def transform(self, data, args, kvargs):
        print("enter UDF StoreEmbedding")

        header = data[0]
        pathIndex = header.index('Path') if 'Path' in header else header.index('path')
        paths = [row[pathIndex].decode('utf-8') for row in data[2:]]

        description_each_path = self._build_descriptions(paths)
        paths_contain_inner_node = list(description_each_path.keys())
        descriptions = list(description_each_path.values())
        with Encoder() as ec:
            embeddings = ec.encode(descriptions)

        dao = MilvusDao(kvargs)
        dao.delete_all()

        insert_results = dao.insert_embeddings(paths_contain_inner_node, embeddings, descriptions)
        success_count = sum([insert_result.succ_count for insert_result in insert_results])
        err_count = sum([insert_result.err_count for insert_result in insert_results])

        dao.load()

        return [
            ["(inserted)", "(failed)"],
            ["INTEGER", "INTEGER"],
            [success_count, err_count]
        ]


# if __name__ == '__main__':
#     udf = UDFStoreEmbedding()
#     paths = [
#         "customer.c_custkey", "customer.c_name", "customer.c_address", "customer.c_nationkey", "customer.c_phone", "customer.c_acctbal", "customer.c_mktsegment", "customer.c_comment",
#         "lineitem.l_orderkey", "lineitem.l_partkey", "lineitem.l_suppkey", "lineitem.l_linenumber", "lineitem.l_quantity", "lineitem.l_extendedprice", "lineitem.l_discount", "lineitem.l_tax", "lineitem.l_returnflag", "lineitem.l_linestatus", "lineitem.l_shipdate", "lineitem.l_commitdate", "lineitem.l_receiptdate", "lineitem.l_shipinstruct", "lineitem.l_shipmode", "lineitem.l_comment",
#         "nation.n_nationkey", "nation.n_name", "nation.n_regionkey", "nation.n_comment",
#         "orders.o_orderkey", "orders.o_custkey", "orders.o_orderstatus", "orders.o_totalprice", "orders.o_orderdate", "orders.o_orderpriority", "orders.o_clerk", "orders.o_shippriority", "orders.o_comment",
#         "part.p_partkey", "part.p_name", "part.p_mfgr", "part.p_brand", "part.p_type", "part.p_size", "part.p_container", "part.p_retailprice", "part.p_comment",
#         "partsupp.ps_partkey", "partsupp.ps_suppkey", "partsupp.ps_availqty", "partsupp.ps_supplycost", "partsupp.ps_comment",
#         "region.r_regionkey", "region.r_name", "region.r_comment",
#         "supplier.s_suppkey", "supplier.s_name", "supplier.s_address", "supplier.s_nationkey", "supplier.s_phone", "supplier.s_acctbal", "supplier.s_comment"
#     ]
#     data = [["path"],["BINARY"]] + [[path.encode("utf-8")] for path in paths]
#     kvargs = {
#         "host": "localhost".encode("utf-8"),
#         "port": "19530".encode("utf-8"),
#     }
#     result = udf.transform(data, None, kvargs)
#     print(result)


class UDFSearchEmbedding:
    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter UDFSearchEmbedding success：", kvargs)
        search_description = kvargs["description"].decode("utf-8")
        pattern = kvargs["pattern"].decode("utf-8")

        search_key_words = LLMDao().provide_keywords(search_description)
        print("key_words:", search_key_words)

        with Encoder() as ec:
            search_embedding = ec.encode([search_key_words])[0]

        similar_paths = MilvusDao(kvargs).search_similarity(search_embedding, pattern)

        return [["(path)"], ['BINARY']] + [
            [path.encode('utf-8')]
            for path in similar_paths
        ]


# if __name__ == '__main__':
#     # 本地测试
#     udf = UDFSearchEmbedding()
#     kvargs = {
#         'port': b'19530',
#         'paths': b'["IvayloIV","Arifyudi26","Jaaga","DevStreet","1804_Apr_USFdotnet","JhonatanGAlves","Alexandre_Caetano_eng","2002_feb24_net","Ellie190","Dalttony","DonnieDing","937447974","AngelesPiotroski","Alfdhiw","Chelsea9803","Alexhendar","654894017","FelipeAN0810","Bingjian_Zhu","GuilhermeOrtizAluno","A_Mckinlay","DannyK1703","Andrew2112","Gamebot2","Ekkyar","DataWorkbench","ICT_BDA","J4Numbers","Girish0212","Aivyss","HarshaVardhan23","EvgeniiZaets","Arthurvdmerwe","Javiithop","AbdelrahmanElghalla","George_Kagwe","DaniloAlmeidaSantos","97lynk","CompassPointMedia","Areso","BigBoi077","Ignis34Rus","3203317","Bhavanshuvig","ESTS_RS","Diegocortes15","GambuzX","HenishPatadiya","Al_Ibne_Siam","Em11FW","0815_edv","ChathurRandul","Byegon2441","Esneider1997","HariShankar08","DirkReinemann","JohnMMMM","JimmyMayta","GitHubRepoDescription","Hasindu1","JacobCreed2","HazarZYGC","AlpetGexh","CS445F21_PACU","BizbrainzGit","Anupam_Panwar","Arc2014","Arvato_Systems","DaviMartinss","BrowserGameScriptz","FatemaBader","Jotinha65","FauzanKatil","Cat_nyan","Abbottmo","AaEzha","GranadaORM","DaffaDwiyanti","JonaCaste","IulianCernat","Bcdo","InsightsDev_dev","Direct_Entry_Program_7_Playground","BallardDavid","CS_UCY_EPL343","Imran_cse","GridGain_Demos","Andytule","DarkCobra7423","DanIulian","BoiseState","Jose_augusto_git","Elzawawy","CoderDream","Kaciras","Chrismond_Versailles","AvnanRahman","Aldirezkir","Cold23","Daniel_Ramos_Garcia","Dania01","Bucknell_ECE","8razel8","FcoJavierGlez","Afifhendrawan_77","AhmedinM","Alfarizqi88","Fernandogza","AlbertMukhammadiev","GustavoOliverRocha","AntonKJ","Cantara","JeremyPercy","Communote","148360","IvanLychkovakha","DayanaChris","Cameron_Weber","BrendaSalgadoCaldera","Feeco5","52North","DefJia","GeorgiGradev","Code4SocialGood","FedulovEgor","AnaghaV19","Jonathan_Roddy","AlissonJF","Andi_IM","CTU_ITClub","IEA_Task_43","AdamArthurF","Cyrrav","Adjagbale_Yao","Brsrker","BBucketIsBetter","1909_sep30_net","Grandez","ImpulseCorona101","AthinaSpanou","Anggifitra141","BitterOcean","Graftiger","AdityaSrinivasa","Bakuard","IGedeMiarta","Jacint56","Eddie_Graham","Frallallero","JabRef","HuuDungNg","Akhyruyatul","DebashishSau","GutherJos","Foroozani","Ddollz","BrunoCampana","BryceDouglasJames","ArneKramerSunderbrink","Api2sem2021","DieuLinh99","Elisee153","AdamNoone","Dr0na","HenriqueBraz","Bastienp2a","JavaZWT","ColinM94","AlisuSantana","CodingBeard","837477","AlekseyBykov","BobyHart4488","2pc","Aureliano1963","GITSALAHE","AlphaWeb1","Cynler","H171600610","Gladsonms","DataScientists","Eynosoft","Coffe_chill","DWIKEIKROMI","ChamaniS","AgladeJesus","CesarAldair123","AXNTROYUANXD","CANSA_team","AndrichardWS","BuildForSDG","JamesKing9","Fadhilamadan","Atihinen","Akbhobhiya","FreddieValenzuela","Ardi_bog","Harsh1925","Devyani1907","G3G4X5X6","FelCore","HackBrexit","DataViva","9606","Denzel18","Islandora_Devops","Jiumiking","Evilscaught","EhODavi","ATetiukhin","Aashishraizada","Greenborn","Dayana20","GabrielSA87","2_men_team","ArieleMartins","BenediktMagnus","CPSC319_2017w1","EquipstatTSEC","2012lucho","Adetiya21","HaidirBz","DavidBarbosa425","6299481145","Alessandro_Schmidt","Devansh3712","AndanTeknomedia","GustavoAT","KaisCommitted","EnvironmentalDashboard","JavierMtzO","Griffin_Brome","AldiAkbar","Dominick159","Debdyut","FeurialBlack","FernandoChai","DhrumilShah98","Fariq01","IngSW_unipv","Ermile","GrzegorzMika","AgileCrocodile","Aiyuuu033119","ChangYeop_Yang","Freakazo","CPuriandika","2binsurranceasmr","Binny29","Boyan_Apostolov","BioAnalyticResource","JoaoJanini","AngryJKirk","AzureKn1ght","GodBastardNeil","EleaFederio","DRIFSRI","Daniel_Tilley","BAMGames","Joseki","9287vk5","AzrulSudarmin","JoergRoemhild","CheungChingYin","CloudPOSFall","AlexandreLch","DoubtAvatar_DP2","DawarAlvi","Anggito28","Alvianrizky","Hetal2425","3m1n3nc3","GokselKUCUKSAHIN","AsciiShell","FreezyBee","3rdYearGroup11","Femeuc","Ashish_003","DuckWithNoSound","BliiTzZ","Akash_Trivedi","BinhMinhs10","DaviJam","B0urG3ois","ChrisAraneo","Didi3aone","IgorIvkin","Ikhlasul_FZ","CharlieGoldsmithAssociates","Frissons","Asmodasis","Aastha2001","AlfianChandra","JohnDoeAntler","JonRob812","476661640","BhagyaRana","Doctor_Hacker","Daviad0","IANSOFT_AC","B_Yan","Dri0m","JavGt","CUAHSI","Gamdara","JesusHdezWaterloo","AbhishekMali21","Barto12","Chris95Hua","FalianaRanai","Aran276","BarrelBrenner","GMCarlos","Carduin","FreddoCG","7cnny","2006_jun15_net","0cmg","Jochen1602","AccaEmme","Arifianto12RPLA","AyaanH123","DXane","Engin_Boot","Benjith","Dissem","CaioEduardoMouta","Jimut123","Isti_Am","Fireserdg","DrWolf_OSS","ErickSantiagoUyana","Furlanetti","Danieloliver11","Cadiac","HairAndBeardGuy","AlphawizzTechnology","IngDixonCano","JoaopedroSassi","Dinara2020","Jugendhackt","BacLuc","JamesMarino","JosenildoMauricio","AdmiralPuni","AirportOs","JhonzRamos","BeyondLogicInc","GregPetropoulos","Anonyymi","AylinArtut","Julio_Antony","Charlene76140","Java_Publications","HeroBarry","Brain2Github","Chief_Ut","ArtyshkoAndrey","GrayXu","747646769","EgaBudimanItera","JrzenonDev","HaSa1002","Dvillano","GydroCasper","CesarCasagrande33","IBARTI2019","Alexhaoge","Anusien","Becold","BlagoKolev","GlistenSTAR","Daryl110","BuffaloShop","ExplosiveBattery","BobSimon","BogdanMarghescu","Hide_Koba","Indah17","4nd12i","DavidGalileo24","HamidXoliqov","H_N41K","CERA_OHM","Gabriel_Blanes","1163710122","AdvenAdam","1lirisist","ImpalaToGo","A_Lorin","AnisaDyah","FlorentGallou_Dev","AlzheTV","CliffordMarley","JamaHCS","Cepave","AssadIKhan","ArturTomasi","BinoMate","CUBRID","AncientMariner","BeiyanLuansheng","2010USFJava","Abhinay_Reddy","AurelieBodart","Cafe_Variome","Apicurio","1ibrary","Abel_Moremi","Arctos6135","Apop85","AlexnaderMishin","Aurelius91","Jupriadi","ComPHPPuebla","Juzzephe","Gfrey70","Jonatas_Soares_Alves","GeNa_jj","DigitalDevelooper","HarshaAbeyvickrama","FaizullahFirozi","Felix_xilef","GustavoBorges_tec","Ivrgs","AkbarMuarif","F4NT0","Fredy_Gutierrez","JonahY","88aleksandra88","JPablo1997","Guavus","Angular2Guy","Alcc5","Alfa93Adv","Ir001","BasedDatabases","344546752","Aquerr","Amuxix","Andy_Merhaut","Jev1337","Clifford18","ASCIT","DeaVenditama","D13xxx","BelmiroMungoi","ElephanZ","HKK_Team","Antonio_Rdz","Chizzy_codes","DenisStolyarov","CMS_Project","FelipePDS","BugFixes","Jiyoung5242","IsabellaTorres100111","BorisKlinkerSAP","ConcaveIT","Cherylngo","BogushAleksandr","Aryan284","Gregseanyoung","JLMadsen","Alphinha","173716414","GayanSampathManamendra","Barraguesh","JordanForde1","JMAfrico","Ebl0010","AyberkCakar","Godeta","Illumiy","CorbenTerminator","AndreaBizz8","FieryInferno","Alachisoft","Egg4","FlashZoom","IgorFroehner","DanielVallin","CSTeam_Squirtle","HouariZegai","FerdinandSukhoi","Enrique213_VP","18502079446","Danangoffic","Alfraganus","HXSecurity","BanzaiTokyo","AirLiquide","GuidoTorres","AriniInf","AnsariMaviya","Danieljrsilva","ATOM27","HarryCordewener","CELEC_USTHB_CLUB","BorjaPelegrin","DaniyanP","AvindaAlamsyah","Dukou007","GabrielGardev","DarlanNoetzold","Fernal73","Darthveloper21","BuntsFidleyBits","Harprit_singh","Harvard_ATG","FalahRafif","EstefaniaExamples","Don_Jin","JonathanGWesterfield","Arnzero","AVE_cesar","AlfanFG","JuanCamiloRB","AnakCreative","GuoHaoZai","Groupe2_Musicoshop","Brayan7u7r","Hafizcode02","EJNB","DOH_CHD_CARAGA","JavaDogs","19Nikola96","Breeze1in1drizzle","250203726","FriendsOfREDAXO","Jeet21_dev","Gempitalarasati","Ithar","EdwinFLopez","Dracenco","IkeC","664709923","FlakyTestDetection","Bartleby2718","DrakeJohnny","HabibAroua","JoeLago","Aresha_RS","HappyLamia","Alch1mist","Aplear","Iteachprogramer","Bohemiaman","AyoubNafil","Guilherme_Sampaio","France_ioi","IgnasiBosch","Embarcadero","JamithNimantha","BenjaminAtbi","Emmett09","BlackCubes","JSRevolorio","E_Arsip","Bishobokeruwizeye","AtlasOfLivingAustralia","Blackpaper13","Alexlingl","Dinesh_Wasnik","HUSTERGS","Dasep12","AshfinRamadhandy","JacobBaynes","GuilhermePalma","CaliiTapia","HuangShubin99","Heroadn","AliHSZ99","HuyCongJr","JadynWong","GabrielEVT","ITMSFT","CodesAreHonest","AscEmu","JAMESKURIA","Algifarii","Ignasrocas1990","Jely101","J04N4","Cassolette","Agwis_Software","DavidHigueraFerrez","KValexander","DanielHenrique_Dev","2504Guimaraes","FerdianPio","Jcarnecer","Galuh80","Emesson_cmd","CristianSalazarAtalaya","DSM_DMS","Imam9","HosseinChibane","Ja3farMortada","Du_an_Giao_Duc","4156Team","GavrielDunev","1071607950","Evodia123456","Dandyamarta211","Fairizal","AndhikaK","JoaoG23","FarhanShoukat","FFahrenheit","Barry0310","ASXFA","AgnieszkaCh","IgorGuariroba","0_k_1","INF2021_PW_G20","IqbalSoft","AngelFlower","5730289021_NN","Black_Library","CodeFuller","GustavoQuinteroC","CareersSkillsIncubator","Activiti","Ankush34","AldonahZero","AliAbdurohman16","Bruuno07","AyuMuhafilah","DimovDimo","HeyCommunity","AndyZunaedy","1612SMShuvo","FarisLucky","BrunoTravassos","AlwinBrauns","AlexanderShniperson","AlanLWilliams","DrewBritt","Camillolevi","AroniainaSaotra","FutureB1t","ASRSoftware","AbsaOSS","Heggy19","BenatG_tech","Infact27","Aamir_97","Juniper","CactuseSecurity","Csineneo","ActiveBeanCoders","CVSink","Harlen520","ElGarageHub","GROUPBAOCAO4305","Gellish","FilanMaulaAndini","Aplycaebous","Braz99","JEMinick","Cauenumo","H_Gallardo","Eberm024","EvanGertis","Bigjoos","Dowsley","Freire71","Fayiawaluddinzaki","310369677","DrunkenLee","Bo_Xuan","AnathPKI","JohnLeather","EtienneCClarke","Ademboussetha","AdoboFighter","DaoDucVTCA","APIJSON","JC_Rave","ChecheSwap","GleysonAndrade","Baboo16cs11","362409960","IrfanFananiM","AnkhSalam","FleetFarming","Juanca92","Edsonrc","Jiachen_Zhang","Checkers300","GagaPoloJr","HigorRoc","AKASH_2019","JMonks14","AbhayLodhi","Danish12","121github","ChiaraDM","Ensembl","Jonasdart","CEPRE_UNI","DongJeremy","Amaulid","AmineMbaye","JWeonseok","BerliozLeChat","DCaceres2018","CurtisPreston99","CembZy","DannyRivasDev","Enter_36_chambers_of_wu_tang_fam","HalfMouse","Dragontalker","FerMdez","AlekseiSkr","AndreMoerza","Andrianto5501","528854302","ChenhuaFan","HVrettost","AhmedGamal98","JManToGithub","Ferriol_blip","8108905324","JakubVazan","Angelica2001143","CyberArkForTheCommunity","HuLing1025","Fandia9050","CareSet","DavidOlivera89","4kari","10go1027","AKS144","KKgautam61","Deluze","ElianMariano","Islan_Santos","AlexeyDota2Ru","ABenchM","AndreaBIGOT","Darkr4ptor","BiancaGiovanna","FoxBPM","ADEPT_Informatique","3plcoins","Excecutor1","Carter90","DataJunction","EndyPratama","CiccioTecchio","AlaaRdwan94","AryanArion","Escapist_007","AlexandreSato","CorentinMAG","Floating_Island","Azad94","JotunMichael","1988gadocansey","EricMalpass","IAAA_Lab","Finnerale","200106_UTA_PRS_NET","FationSH","Aiiishaaa","Gabo1122","09143613","AlekseevArtem","BrenoHenrrique","JosePerezHG","EdwinKassier","Fly_76","DenislavVelichkov","JONAS060708","Azzam279","AlejandroSantorum","HXLStandard","JoaquinMachXD2021","Dumbeldor","1809_UTA_Java","K4M1coder","Haikson","Dafana29","EghoPratama"]',
#         'host': b'localhost',
#         'description': b'value'}
#     result = udf.transform(None, None, kvargs)
#     print(result)


class UDFAnalyseRelation:
    RESULT_HEADER = [
        ["(source)", "(target)", "(score)", "(description)"],
        ['BINARY', 'BINARY', 'DOUBLE', 'BINARY'],
    ]

    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter UDFAnalyseRelation success:", kvargs)
        sources_json = kvargs["sources"].decode("utf-8")
        targets_json = kvargs["targets"].decode("utf-8")

        dao = MilvusDao(kvargs)
        source_result = list(dao.fetch_by_path(sources_json))
        print("source_result:", len(source_result))

        if not source_result:
            return self.RESULT_HEADER

        source_paths, source_embeddings, source_descriptions = (list(t) for t in zip(*source_result))

        target_result = dao.bulk_search_similarity(source_embeddings, targets_json)
        relations = [
            (source_path, target_path, similarity, source_description, target_description)
            for i, (source_path, source_description) in enumerate(zip(source_paths, source_descriptions))
            for target_path, target_description, similarity in target_result[i]
        ]

        print("relations:", len(relations))

        if not relations:
            return self.RESULT_HEADER

        # Create a DataFrame from the relations list with columns: source, target, and similarity
        df = pd.DataFrame(relations,
                          columns=["source", "target", "similarity", "source_description", "target_description"])

        # Extract the root part (before the first dot) from source and target columns
        df['source_root'] = df['source'].str.split('.').str[0]
        df['target_root'] = df['target'].str.split('.').str[0]

        # Filter out records where source_root and target_root are the same
        df = df[df['source_root'] != df['target_root']]

        # Find the maximum similarity for each unique pair of source_root and target_root
        max_similarities = df.loc[
            df.groupby([df[['source_root', 'target_root']].apply(frozenset, axis=1)])['similarity'].idxmax()]

        # Select only the source, target, and similarity columns from the DataFrame
        relations = max_similarities[['source', 'target', 'similarity', 'source_description', 'target_description']]

        # Select top 1/3 of the most relations based on similarity
        limit = 3
        relations = relations.sort_values(by='similarity', ascending=False).head(limit)

        if len(relations) == 0:
            return self.RESULT_HEADER

        # 并行获取 source_description 和 target_description 的关系的描述
        relations['description'] = LLMDao().describe_relation(
            list(zip(relations['source_description'], relations['target_description'])))
        relations = relations[['source', 'target', 'similarity', 'description']]

        # Encode the source and target columns as UTF-8
        relations.loc[:, 'source'] = relations['source'].str.encode('utf-8')
        relations.loc[:, 'target'] = relations['target'].str.encode('utf-8')
        relations.loc[:, 'description'] = relations['description'].str.encode('utf-8')

        # Return the final result as a list of lists, including headers and data types
        return self.RESULT_HEADER + relations.values.tolist()


# if __name__ == '__main__':
#     # 本地测试
#     udf = UDFAnalyseRelation()
#     kvargs = {'sources': b'["1612SMShuvo_sub_contract","2010USFJava_MaurerP","1lirisist_kanbank"]', 'port': b'19530', 'targets': b'["1lirisist_kanbank","1ibrary_1ibrary_back_end"]'}
#     result = udf.transform(None, None, kvargs)
#     print(result)


class Node(NamedTuple):
    path: str
    description: str
    embedding: np.array(np.float32)
    children: list['Node'] = []


class Aggregator:
    def __init__(self, threshold=0.8, n_clusters=8, pca_components=50, tsne_components=3):
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.tsne_components = tsne_components

    def _build_cluster_labels(self, nodes: list[Node], cluster_target) -> np.array(np.int32):
        print("Building cluster for", len(nodes), "nodes")

        embeddings = [node.embedding for node in nodes]

        km = KMeans(n_clusters=cluster_target, random_state=0)
        km.fit(embeddings)
        return km.labels_

    def _nest_build_cluster(self, leaf_nodes: list[Node], target: int, depth: int) -> list[Node]:
        cluster_target = target ** depth

        if cluster_target >= len(leaf_nodes):
            print("Reached target depth", depth)
            return leaf_nodes

        print("Building nested cluster for", len(leaf_nodes), "nodes, target:", target)
        children_nodes = self._nest_build_cluster(leaf_nodes, target, depth + 1)

        labels = self._build_cluster_labels(children_nodes, cluster_target)

        clusters = defaultdict(list)
        for label, node in zip(labels, children_nodes):
            clusters[label].append(node)
        children_list = list(clusters.values())
        children_path_clusters = [[node.path for node in cluster] for cluster in children_list]
        children_description_clusters = [ [node.description for node in cluster] for cluster in children_list]

        current_paths = LLMDao().summarize(children_description_clusters)
        current_descriptions = [
            ", ".join([current_path] + children_paths_cluster)
            for current_path, children_paths_cluster in zip(current_paths, children_path_clusters)
        ]

        with Encoder() as ec:
            current_embeddings = ec.encode(current_descriptions)
        return [
            Node(path, description, embedding, children)
            for path, description, embedding, children in
            zip(current_paths, current_descriptions, current_embeddings, children_list)
        ]

    def build_cluster(self, leaf_nodes: list[Node], target: int) -> list[Node]:
        return self._nest_build_cluster(leaf_nodes, target, 1)


class UDFMerge:

    def _generate_result(self, nodes: list[Node], parent_path: list[str], results: list[list]):
        for node in nodes:
            if not node.children:
                results.append([node.path, ".".join(parent_path)])
            else:
                parent_path.append(node.path)
                self._generate_result(node.children, parent_path, results)
                parent_path.pop()

    def transform(self, data, args, kvargs):
        print("enter transform Merge success:", kvargs)
        paths_json = kvargs["paths"].decode("utf-8")

        paths_result = MilvusDao(kvargs).fetch_by_path(paths_json)
        nodes = [Node(path, description, embedding) for path, embedding, description in paths_result]
        nested_clustered_nodes = Aggregator().build_cluster(nodes, 18)

        results = [["(name)", "(cluster)"], ['BINARY', 'BINARY']]
        self._generate_result(nested_clustered_nodes, [], results)
        print("results:", results)
        return results

# if __name__ == '__main__':
#     # 本地测试
#     udf = UDFMerge()
#     kvargs = {
#         'paths': b'["2002_feb24_net_diana_code","0cmg_Mybatis_PageHelper","2006_jun15_net_pamela_code","1163710122_hit_1163710122","1lirisist_kanbank","19Nikola96_les_petits_paniers","1988gadocansey_pos_codeigniter","250203726_ams","1804_Apr_USFdotnet_asanjose_project0","1071607950_phonemall","200106_UTA_PRS_NET_P0_Therese_Parks","1612SMShuvo_sub_contract","09143613_MS","0_k_1_BlogDjango","2010USFJava_MaurerP","1909_sep30_net_javon_project0","0815_edv_Bierkasse","2006_jun15_net_william_code","1809_UTA_Java_project_0_Darchie28","121github_121it","2504Guimaraes_Trabalho_CRUD_Springboot","2006_jun15_net_noah_code","344546752_mybatis","2002_feb24_net_stacey_project0","148360_resultkpis","173716414_TeamB2","2binsurranceasmr_BookStorePhp","10go1027_sistemaLavaderoPp3","2002_feb24_net_shawn_code","2010USFJava_MoonberryP1","2012lucho_OpenPriceStadisticsBack","2006_jun15_net_kirti_code","3203317_yb","310369677_ycd_framework","18502079446_cusss","2_men_team_components_labs","2pc_hive_3","2006_jun15_net_daniel_code","1ibrary_1ibrary_back_end","1909_sep30_net_jose_code"]'    }
#     result = udf.transform(None, None, kvargs)
#     print(result)
