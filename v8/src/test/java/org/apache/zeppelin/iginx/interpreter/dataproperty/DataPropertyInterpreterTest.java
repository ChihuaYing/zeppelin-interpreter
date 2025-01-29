package org.apache.zeppelin.iginx.interpreter.dataproperty;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.apache.velocity.VelocityContext;
import org.apache.zeppelin.iginx.util.VelocityUtil;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class DataPropertyInterpreterTest {
  private DataPropertyInterpreter interpreter;

  @BeforeEach
  void setUp() {
    interpreter = new DataPropertyInterpreter(null, null, 0);
  }

  @Test
  void testRenderDataProperty() throws IOException {
    List<String[]> paths = getTpchPaths();
    InterpreterContext context = getInterpreterContext();

    // Call the method to render data property
    String html = interpreter.generateDataPropertyHtml(paths, context);

    // 将结果写入临时文件并将路径打印出来
    Path path = Files.createTempFile("data_property", ".html");
    Files.write(path, html.getBytes());
    System.out.println("Data property html file path: " + path.toUri());
  }

  @Test
  void testRenderDataPropertyGraph() throws IOException {

    VelocityContext velocityContext = new VelocityContext();
    velocityContext.put("paragraphId", "testParagraphId");
    velocityContext.put("nodeList", getNodeListString());
    velocityContext.put("relationList", getRelationListString());
    String html = VelocityUtil.generate("templates/data-property.vm", velocityContext);

    // 将结果写入临时文件并将路径打印出来
    Path path = Files.createTempFile("data_property_graph", ".html");
    Files.write(path, html.getBytes());
    System.out.println("Data property graph html file path: " + path.toUri());
  }

  private List<String[]> getTpchPaths() {
    return Arrays.asList(
        new String[] {"customer", "c_acctbal"},
        new String[] {"customer", "c_address"},
        new String[] {"customer", "c_comment"},
        new String[] {"customer", "c_custkey"},
        new String[] {"customer", "c_mktsegment"},
        new String[] {"customer", "c_name"},
        new String[] {"customer", "c_nationkey"},
        new String[] {"customer", "c_phone"},
        new String[] {"lineitem", "l_comment"},
        new String[] {"lineitem", "l_commitdate"},
        new String[] {"lineitem", "l_discount"},
        new String[] {"lineitem", "l_extendedprice"},
        new String[] {"lineitem", "l_linenumber"},
        new String[] {"lineitem", "l_linestatus"},
        new String[] {"lineitem", "l_orderkey"},
        new String[] {"lineitem", "l_partkey"},
        new String[] {"lineitem", "l_quantity"},
        new String[] {"lineitem", "l_receiptdate"},
        new String[] {"lineitem", "l_returnflag"},
        new String[] {"lineitem", "l_shipdate"},
        new String[] {"lineitem", "l_shipinstruct"},
        new String[] {"lineitem", "l_shipmode"},
        new String[] {"lineitem", "l_suppkey"},
        new String[] {"lineitem", "l_tax"},
        new String[] {"nation", "n_comment"},
        new String[] {"nation", "n_name"},
        new String[] {"nation", "n_nationkey"},
        new String[] {"nation", "n_regionkey"},
        new String[] {"orders", "o_clerk"},
        new String[] {"orders", "o_comment"},
        new String[] {"orders", "o_custkey"},
        new String[] {"orders", "o_orderdate"},
        new String[] {"orders", "o_orderkey"},
        new String[] {"orders", "o_orderpriority"},
        new String[] {"orders", "o_orderstatus"},
        new String[] {"orders", "o_shippriority"},
        new String[] {"orders", "o_totalprice"},
        new String[] {"part", "p_brand"},
        new String[] {"part", "p_comment"},
        new String[] {"part", "p_container"},
        new String[] {"part", "p_mfgr"},
        new String[] {"part", "p_name"},
        new String[] {"part", "p_partkey"},
        new String[] {"part", "p_retailprice"},
        new String[] {"part", "p_size"},
        new String[] {"part", "p_type"},
        new String[] {"partsupp", "ps_availqty"},
        new String[] {"partsupp", "ps_comment"},
        new String[] {"partsupp", "ps_partkey"},
        new String[] {"partsupp", "ps_suppkey"},
        new String[] {"partsupp", "ps_supplycost"},
        new String[] {"region", "r_comment"},
        new String[] {"region", "r_name"},
        new String[] {"region", "r_regionkey"},
        new String[] {"supplier", "s_acctbal"},
        new String[] {"supplier", "s_address"},
        new String[] {"supplier", "s_comment"},
        new String[] {"supplier", "s_name"},
        new String[] {"supplier", "s_nationkey"},
        new String[] {"supplier", "s_phone"},
        new String[] {"supplier", "s_suppkey"});
  }

  private InterpreterContext getInterpreterContext() {
    return new InterpreterContext(
        "noteId",
        "paragraphId",
        "replName",
        "paragraphTitle",
        "paragraphText",
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null);
  }

  private String getNodeListString() {
    return "[{\"depth\":0,\"id\":\"rootId\",\"name\":\"Data Asset\"},{\"depth\":1,\"id\":\"rootId.Diverse_software_projects_and_tools_for_various_applications_including_databases_web_development_gaming_and_system_management\",\"name\":\"Diverse software projects and tools for various applications including databases web development gaming and system management\"},{\"depth\":1,\"id\":\"rootId.Diverse_database_and_web_development_projects_by_multiple_contributors_covering_SQL_practice_educational_tools_personal_projects_and_various_tech_applications\",\"name\":\"Diverse database and web development projects by multiple contributors covering SQL practice educational tools personal projects and various tech applications\"},{\"depth\":1,\"id\":\"rootId.Diverse_software_and_project_development_including_management_systems_databases_applications_APIs_web_tools_educational_resources_game_development_and_open-source_collaborations\",\"name\":\"Diverse software and project development including management systems databases applications APIs web tools educational resources game development and open-source collaborations\"},{\"depth\":1,\"id\":\"rootId.Open-source_and_collaborative_software_projects_across_diverse_domains_including_security_data_management_IoT_backend_development_healthcare_microservices_server_management_and_tech_frameworks_with_various_APIs_and_system_updates\",\"name\":\"Open-source and collaborative software projects across diverse domains including security data management IoT backend development healthcare microservices server management and tech frameworks with various APIs and system updates\"},{\"depth\":1,\"id\":\"rootId.Hotel_and_travel_management_systems_development_including_booking_and_data_platforms_by_various_developers\",\"name\":\"Hotel and travel management systems development including booking and data platforms by various developers\"},{\"depth\":1,\"id\":\"rootId.Online_users_engage_in_diverse_tech_projects_share_interests_and_develop_web_applications_across_various_domains\",\"name\":\"Online users engage in diverse tech projects share interests and develop web applications across various domains\"},{\"depth\":1,\"id\":\"rootId.Hive-based_components_project_management_tools_web_frameworks_cloud_services_data_platforms_developer_contributions_and_various_software_projects_including_security_backup_and_database_solutions\",\"name\":\"Hive-based components project management tools web frameworks cloud services data platforms developer contributions and various software projects including security backup and database solutions\"},{\"depth\":1,\"id\":\"rootId.Educational_and_professional_projects_involving_various_systems_applications_and_research_by_students_and_developers_across_different_domains\",\"name\":\"Educational and professional projects involving various systems applications and research by students and developers across different domains\"},{\"depth\":1,\"id\":\"rootId.Diverse_project_identifiers_across_software_gaming_development_analytics_and_platforms_involving_various_creators_and_technical_domains\",\"name\":\"Diverse project identifiers across software gaming development analytics and platforms involving various creators and technical domains\"},{\"depth\":1,\"id\":\"rootId.Diverse_web_projects_and_blogs_by_various_developers_using_multiple_technologies_and_frameworks_including_Django_Laravel_WordPress_PHP_Vue_and_covering_personal_team_university_and_social_media_platforms\",\"name\":\"Diverse web projects and blogs by various developers using multiple technologies and frameworks including Django Laravel WordPress PHP Vue and covering personal team university and social media platforms\"},{\"depth\":1,\"id\":\"rootId.Diverse_software_and_web_development_projects_across_various_domains_and_technologies_by_multiple_developers_including_educational_tools_CMS_systems_e-commerce_gaming_databases_AI_chatbots_project_management_tools_and_specialized_applications\",\"name\":\"Diverse software and web development projects across various domains and technologies by multiple developers including educational tools CMS systems e-commerce gaming databases AI chatbots project management tools and specialized applications\"},{\"depth\":1,\"id\":\"rootId.Daniel_and_Pamela's_2006_net_codes_to_child_vaccine_app_development_and_various_tech_projects_by_multiple_individuals\",\"name\":\"Daniel and Pamela's 2006 net codes to child vaccine app development and various tech projects by multiple individuals\"},{\"depth\":1,\"id\":\"rootId.Database_and_SQL_education_projects_across_languages_and_domains_including_developer_contributions_and_skill_enhancement_resources\",\"name\":\"Database and SQL education projects across languages and domains including developer contributions and skill enhancement resources\"},{\"depth\":1,\"id\":\"rootId.Diverse_coding_projects_and_resources_from_students_professionals_and_institutions_involving_various_frameworks_and_applications\",\"name\":\"Diverse coding projects and resources from students professionals and institutions involving various frameworks and applications\"},{\"depth\":1,\"id\":\"rootId.Coding_and_database_projects_across_industries_including_web_development_educational_platforms_veterinary_systems_and_diverse_applications_using_SQL_PHP_Springboot_Angular_Java_EE_and_other_technologies\",\"name\":\"Coding and database projects across industries including web development educational platforms veterinary systems and diverse applications using SQL PHP Springboot Angular Java EE and other technologies\"},{\"depth\":1,\"id\":\"rootId.Developers_creating_diverse_projects_in_Java_web_development_databases_IoT_Docker_and_various_technologies_including_educational_tools_andFlowable_engine_usage_for_applications_and_systems\",\"name\":\"Developers creating diverse projects in Java web development databases IoT Docker and various technologies including educational tools andFlowable engine usage for applications and systems\"},{\"depth\":1,\"id\":\"rootId.Diverse_web_software_and_e-commerce_projects_spanning_retail_healthcare_inventory_management_systems_and_personal_educational_coding_initiatives_involving_various_technologies_and_platforms\",\"name\":\"Diverse web software and e-commerce projects spanning retail healthcare inventory management systems and personal educational coding initiatives involving various technologies and platforms\"},{\"depth\":1,\"id\":\"rootId.Diverse_individual_and_collaborative_projects_in_fields_like_database_studies_software_development_smart_homes_epidemiology_and_more_by_various_developers_including_specific_usernames_and_topics\",\"name\":\"Diverse individual and collaborative projects in fields like database studies software development smart homes epidemiology and more by various developers including specific usernames and topics\"}]";
  }

  //  private String getRelationListString() {
  //    return "[{\"depth\":0,\"id\":\"rootId\",\"name\":\"Data
  // Asset\"},{\"depth\":1,\"id\":\"rootId.Diverse_software_projects_and_tools_for_various_applications_including_databases_web_development_gaming_and_system_management\",\"name\":\"Diverse software projects and tools for various applications including databases web development gaming and system management\"},{\"depth\":1,\"id\":\"rootId.Diverse_database_and_web_development_projects_by_multiple_contributors_covering_SQL_practice_educational_tools_personal_projects_and_various_tech_applications\",\"name\":\"Diverse database and web development projects by multiple contributors covering SQL practice educational tools personal projects and various tech applications\"},{\"depth\":1,\"id\":\"rootId.Diverse_software_and_project_development_including_management_systems_databases_applications_APIs_web_tools_educational_resources_game_development_and_open-source_collaborations\",\"name\":\"Diverse software and project development including management systems databases applications APIs web tools educational resources game development and open-source collaborations\"},{\"depth\":1,\"id\":\"rootId.Open-source_and_collaborative_software_projects_across_diverse_domains_including_security_data_management_IoT_backend_development_healthcare_microservices_server_management_and_tech_frameworks_with_various_APIs_and_system_updates\",\"name\":\"Open-source and collaborative software projects across diverse domains including security data management IoT backend development healthcare microservices server management and tech frameworks with various APIs and system updates\"},{\"depth\":1,\"id\":\"rootId.Hotel_and_travel_management_systems_development_including_booking_and_data_platforms_by_various_developers\",\"name\":\"Hotel and travel management systems development including booking and data platforms by various developers\"},{\"depth\":1,\"id\":\"rootId.Online_users_engage_in_diverse_tech_projects_share_interests_and_develop_web_applications_across_various_domains\",\"name\":\"Online users engage in diverse tech projects share interests and develop web applications across various domains\"}]";
  //  }

  private String getRelationListString() {
    return "[]";
  }
}
