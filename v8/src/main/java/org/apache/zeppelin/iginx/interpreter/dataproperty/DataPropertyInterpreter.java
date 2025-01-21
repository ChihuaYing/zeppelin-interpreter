package org.apache.zeppelin.iginx.interpreter.dataproperty;

import cn.edu.tsinghua.iginx.session.Session;
import cn.edu.tsinghua.iginx.session.SessionExecuteSqlResult;
import cn.edu.tsinghua.iginx.utils.FormatUtils;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.*;
import org.apache.velocity.VelocityContext;
import org.apache.zeppelin.iginx.interpreter.dataproperty.entry.GraphData;
import org.apache.zeppelin.iginx.service.NetworkService;
import org.apache.zeppelin.iginx.util.TemplateUtil;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterResult;
import org.apache.zeppelin.interpreter.InterpreterResultMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataPropertyInterpreter {

  private static final Logger LOGGER = LoggerFactory.getLogger(DataPropertyInterpreter.class);
  private static final ObjectMapper MAPPER = new ObjectMapper();

  public enum Config {
    GRAPHICAL_RESULTS("data.property"),
    GRAPHICAL_MERGE("data.property.merge"),
    GRAPHICAL_RELATION("data.property.relation"),
    GRAPHICAL_MERGE_RELATION("data.property.merge.relation"),
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
      InterpreterResult interpreteRresult)
      throws JsonProcessingException {
    switch (sqlResult.getSqlType()) {
      case ShowColumns:
      case Query:
        List<List<String>> queryList =
            sqlResult.getResultInList(false, FormatUtils.DEFAULT_TIME_FORMAT, null);
        List<String> paths = parsePaths(sqlResult);
        if (Config.GRAPHICAL_TREE.isActivated(context)) {
          interpreteRresult.add(
              new InterpreterResultMessage(
                  InterpreterResult.Type.HTML, generateDataPropertyHtml(paths, context)));
        }
        if (Config.GRAPHICAL_RESULTS.isActivated(context)
            || Config.GRAPHICAL_GRAPH.isActivated(context)
            || Config.GRAPHICAL_MERGE.isActivated(context)
            || Config.GRAPHICAL_RELATION.isActivated(context)
            || Config.GRAPHICAL_MERGE_RELATION.isActivated(context)) {
          interpreteRresult.add(
              new InterpreterResultMessage(
                  InterpreterResult.Type.HTML, buildNetworkForShowColumns(queryList, context)));
        }
        break;
      default:
        break;
    }
  }

  private static List<String> parsePaths(SessionExecuteSqlResult sqlResult) {
    List<List<String>> queryList =
        sqlResult.getResultInList(false, FormatUtils.DEFAULT_TIME_FORMAT, null);
    int pathColumnIndex = queryList.get(0).indexOf("Path");
    List<String> paths = new ArrayList<>();
    for (int i = 1; i < queryList.size(); i++) {
      paths.add(queryList.get(i).get(pathColumnIndex));
    }
    return paths;
  }

  String generateDataPropertyHtml(List<String> paths, InterpreterContext context)
      throws JsonProcessingException {
    GraphData.Builder builder = new GraphData.Builder();
    for (String path : paths) {
      builder.addNode(path.split("\\."));
    }
    GraphData graphData = builder.build();
    String graphDataJson = MAPPER.writeValueAsString(graphData);

    VelocityContext velocityContext = new VelocityContext();
    velocityContext.put("paragraphId", context.getParagraphId());
    velocityContext.put("data", graphDataJson);

    return TemplateUtil.generate("templates/data-property.vm", velocityContext);
  }

  public String buildNetworkForShowColumns(
      List<List<String>> queryList, InterpreterContext context) {
    NetworkService networkService =
        new NetworkService(
            Config.GRAPHICAL_MERGE.isActivated(context)
                || Config.GRAPHICAL_MERGE_RELATION.isActivated(context),
            Config.GRAPHICAL_RELATION.isActivated(context)
                || Config.GRAPHICAL_MERGE_RELATION.isActivated(context),
            context.getParagraphId(),
            queryList,
            session,
            milvusHost,
            milvusPort);
    networkMap.put(context.getParagraphId(), networkService);

    VelocityContext velocityContext = new VelocityContext();
    velocityContext.put("paragraphId", context.getParagraphId());
    velocityContext.put("noteId", context.getNoteId());

    return networkService.initNetwork(velocityContext);
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
    context.getConfig().put("needAddHideResult", false);
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
