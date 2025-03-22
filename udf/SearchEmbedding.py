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
        self.key = "204a3ea9bf39f18dd9bf32c71ecbb607.mITgz6pgV7Hzj27A"
        self.model = "GLM-4-Flash"

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


class Encoder:
    def __init__(self, cache_path="embedding_cache.db"):
        print("Initializing Sentence Transformer")
        self.encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
        self.batch_size = 128
        self.cache_path = cache_path
        self.child_weight = 0.3

    def encode(self, descriptions: list[str]) -> list[np.array]:
        print("Calculating embeddings of", len(descriptions), "descriptions")

        # 打开 shelve 缓存
        with shelve.open(self.cache_path) as cache:
            # 找出未缓存的描述
            uncached_descriptions = [desc for desc in descriptions if desc not in cache]
            print("Uncached descriptions:", len(uncached_descriptions))

            # 计算未缓存的 embeddings
            if uncached_descriptions:
                description_batches = [
                    uncached_descriptions[i:i + self.batch_size]
                    for i in range(0, len(uncached_descriptions), self.batch_size)
                ]

                with tqdm.tqdm(total=len(uncached_descriptions), desc="Calculating embeddings") as pbar:
                    for description_batch in description_batches:
                        embedding_batch = self.encoder.encode(description_batch)
                        for desc, embedding in zip(description_batch, embedding_batch):
                            cache[desc] = embedding  # 存入缓存
                        cache.sync()  # 立即写入
                        pbar.update(len(description_batch))

            # 读取缓存
            embeddings = [cache[desc] for desc in descriptions]

        return embeddings


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

    def load(self):
        print("Loading milvus collection")
        self.collection.load()

    def search_similarity(self, embedding, path_list_json):
        print("Searching similar embeddings")

        entities = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            output_fields=["path"],
            limit=10,
            param={"metric_type": "COSINE"},
            expr="path in " + path_list_json
        )

        return [hit.fields["path"] for hit in entities[0]]


class UDFSearchEmbedding:
    def __init__(self):
        pass

    def transform(self, data, args, kvargs):
        print("enter UDFSearchEmbedding success：", kvargs)
        search_description = kvargs["description"].decode("utf-8")
        paths_json = kvargs["paths"].decode("utf-8")

        search_key_words = LLMDao().provide_keywords(search_description)
        print("key_words:", search_key_words)

        ec = Encoder()  # 直接创建实例
        search_embedding = ec.encode([search_key_words])[0]

        similar_paths = MilvusDao(kvargs).search_similarity(search_embedding, paths_json)

        return [["(path)"], ['BINARY']] + [
            [path.encode('utf-8')]
            for path in similar_paths
        ]


class Node(NamedTuple):
    path: str
    description: str
    embedding: np.ndarray
    children: list['Node'] = []


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
