package org.apache.zeppelin.iginx.interpreter.dataproperty;

import cn.edu.tsinghua.iginx.session.Session;
import cn.edu.tsinghua.iginx.session.SessionExecuteSqlResult;
import cn.edu.tsinghua.iginx.utils.FormatUtils;
import com.alibaba.fastjson2.JSON;
import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.apache.zeppelin.iginx.service.NetworkService;
import org.apache.zeppelin.iginx.util.FileUtil;
import org.apache.zeppelin.iginx.util.HighchartsTreeNode;
import org.apache.zeppelin.iginx.util.MultiwayTree;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterResult;
import org.apache.zeppelin.interpreter.InterpreterResultMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class DataPropertyInterpreter {

  private static final Logger LOGGER = LoggerFactory.getLogger(DataPropertyInterpreter.class);

  public enum Config {
    GRAPHICAL_RESULTS("data.property"),
    GRAPHICAL_MERGE("data.property.merge"),
    GRAPHICAL_RELATION("data.property.relation"),
    GRAPHICAL_GRAPH("data.property.graph"),
    GRAPHICAL_TREE("data.property.tree");

    private final String config;

    Config(String config) {
      this.config = ">" + config;
    }

    public String getConfigName() {
      return config;
    }

    public boolean isActivated(InterpreterContext context) {
      return Boolean.parseBoolean((String) context.getConfig().get(config));
    }
  }

  private final Map<String, NetworkService> networkMap = new HashMap<>();

  private Session session;
  private String milvusHost;
  private int milvusPort;
  private String outfileDir;

  public DataPropertyInterpreter(
      Session session, String milvusHost, int milvusPort, String outfileDir) {
    this.milvusHost = milvusHost;
    this.milvusPort = milvusPort;
    this.session = session;
  }

  public boolean canInterpret(String sql, InterpreterContext context) {
    return isHandelHtmlNodeClick(sql) || isHandelHtmlClearParagraph(sql);
  }

  public InterpreterResult interpret(String sql, InterpreterContext context) {
    if (isHandelHtmlNodeClick(sql)) {
      return processHandelHtmlNodeClick(sql, context);
    } else if (isHandelHtmlClearParagraph(sql)) {
      return processHandelHtmlClearParagraph();
    }
    return null;
  }

  public void postProcess(
      InterpreterContext context,
      SessionExecuteSqlResult sqlResult,
      InterpreterResult interpreteRresult) {
    switch (sqlResult.getSqlType()) {
      case ShowColumns:
      case Query:
        List<List<String>> queryList =
            sqlResult.getResultInList(false, FormatUtils.DEFAULT_TIME_FORMAT, null);
        if (Config.GRAPHICAL_TREE.isActivated(context)) {
          interpreteRresult.add(
              new InterpreterResultMessage(
                  InterpreterResult.Type.HTML,
                  buildTreeForShowColumns(queryList, context.getParagraphId())));
        }
        if (Config.GRAPHICAL_RESULTS.isActivated(context)
            || Config.GRAPHICAL_GRAPH.isActivated(context)) {
          interpreteRresult.add(
              new InterpreterResultMessage(
                  InterpreterResult.Type.HTML, buildNetworkForShowColumns(queryList, context)));
        }
        break;
      default:
        break;
    }
  }

  public String buildNetworkForShowColumns(
      List<List<String>> queryList, InterpreterContext context) {
    NetworkService networkService =
        new NetworkService(
            Config.GRAPHICAL_MERGE.isActivated(context),
            Config.GRAPHICAL_RELATION.isActivated(context),
            context.getParagraphId(),
            queryList,
            session,
            milvusHost,
            milvusPort);
    networkMap.put(context.getParagraphId(), networkService);

    String serverAddr = "localhost";
    String serverPort = "8080";
    try {
      LOGGER.info("Current working directory: " + System.getProperty("user.dir"));
      String currentDir = System.getProperty("user.dir");
      File configFile = new File(currentDir, "../conf/zeppelin-site.xml");
      DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
      DocumentBuilder builder = factory.newDocumentBuilder();
      Document document = builder.parse(configFile);
      NodeList propertyList = document.getElementsByTagName("property");
      for (int i = 0; i < propertyList.getLength(); i++) {
        Node propertyNode = propertyList.item(i);
        if (propertyNode.getNodeType() == Node.ELEMENT_NODE) {
          Element propertyElement = (Element) propertyNode;
          String name = propertyElement.getElementsByTagName("name").item(0).getTextContent();
          if ("zeppelin.server.addr".equals(name)) {
            serverAddr = propertyElement.getElementsByTagName("value").item(0).getTextContent();
          } else if ("zeppelin.server.port".equals(name)) {
            serverPort = propertyElement.getElementsByTagName("value").item(0).getTextContent();
          }
        }
      }
    } catch (Exception e) {
      LOGGER.error("Failed to read or parse the Zeppelin configuration file.", e);
    }

    String html =
        networkService
            .initNetwork()
            .replace("PARAGRAPH_ID", context.getParagraphId())
            .replace("NOTE_ID", context.getNoteId())
            .replace("ZEPPELIN_SERVER", serverAddr)
            .replace("ZEPPELIN_PORT", serverPort);

    //    String filePath = "D:\\test.html";
    //    File file = new File(filePath);
    //    try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
    //      writer.write(html);
    //      LOGGER.info("HTML content has been written to {}", filePath);
    //    } catch (IOException e) {
    //      LOGGER.info("Error writing to file: {}", e.getMessage());
    //    }

    return html;
  }

  /**
   * 为show columns 命令创建树状状图
   *
   * @param queryList
   */
  public String buildTreeForShowColumns(List<List<String>> queryList, String paragraphId) {
    MultiwayTree tree = MultiwayTree.getMultiwayTree();
    queryList
        .subList(1, queryList.size())
        .forEach(
            row -> {
              MultiwayTree.addTreeNodeFromString(tree, row.get(0));
            });
    String htmlTemplate = "static/highcharts/tree.html";
    try (InputStream inputStream =
        this.getClass().getClassLoader().getResourceAsStream(htmlTemplate)) {
      BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
      StringBuilder content = new StringBuilder();
      String line;
      while ((line = reader.readLine()) != null) {
        content.append(line).append("\n");
      }
      List<HighchartsTreeNode> nodeList = new ArrayList<>();
      int depth = tree.traverseToHighchartsTreeNodes(tree.getRoot(), nodeList);

      String jsonString = JSON.toJSONString(nodeList);
      String html =
          content
              .toString()
              .replace("PARAGRAPH_ID", paragraphId)
              .replace("NODE_LIST", jsonString)
              .replace("TREE_DEPTH", String.valueOf(depth))
              .replace("TREE_ENABLE", String.valueOf(true));
      String fileName = paragraphId + "_tree.html";
      // 写入文件服务器paragraphID_tree.html
      String targetPath = outfileDir + "/graphs/tree/" + fileName;
      FileUtil.writeFile(html, targetPath);
      return html;
    } catch (IOException e) {
      LOGGER.warn("load show columns to tree error", e);
    }
    return "";
  }

  private static boolean isHandelHtmlNodeClick(String sql) {
    return sql.startsWith("handle_html_node_click");
  }

  private InterpreterResult processHandelHtmlNodeClick(String sql, InterpreterContext context) {
    LOGGER.info("enter processHandelHtmlNodeClick, the sql is {}", sql);
    InterpreterResult interpreterResult;
    String[] strings = sql.split(" ");
    String paragraphId = strings[1];
    String nodeId = strings[2];
    NetworkService networkService = networkMap.get(paragraphId);
    if (networkService == null) {
      interpreterResult =
          new InterpreterResult(
              InterpreterResult.Code.ERROR,
              "the networkService with paragraphId: " + paragraphId + " is null");
      return interpreterResult;
    }
    String msg = networkService.handleNodeClick(nodeId);
    LOGGER.info("processHandelHtmlNodeClick: msg is {}", msg);

    interpreterResult = new InterpreterResult(InterpreterResult.Code.SUCCESS);
    interpreterResult.add(InterpreterResult.Type.TEXT, msg);
    context.getConfig().put("needAddHideResult", true);
    return interpreterResult;
  }

  private static boolean isHandelHtmlClearParagraph(String sql) {
    return sql.equals("clearParagraph");
  }

  private InterpreterResult processHandelHtmlClearParagraph() {
    InterpreterResult interpreterResult = new InterpreterResult(InterpreterResult.Code.SUCCESS);
    interpreterResult.add(InterpreterResult.Type.TEXT, "");
    return interpreterResult;
  }
}
