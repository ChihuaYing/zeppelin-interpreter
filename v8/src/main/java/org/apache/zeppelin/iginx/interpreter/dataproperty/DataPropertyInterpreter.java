package org.apache.zeppelin.iginx.interpreter.dataproperty;

import cn.edu.tsinghua.iginx.exception.SessionException;
import cn.edu.tsinghua.iginx.session.Session;
import cn.edu.tsinghua.iginx.session.SessionExecuteSqlResult;
import cn.edu.tsinghua.iginx.utils.FormatUtils;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Preconditions;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.velocity.VelocityContext;
import org.apache.zeppelin.iginx.interpreter.dataproperty.entry.GraphData;
import org.apache.zeppelin.iginx.service.NetworkService;
import org.apache.zeppelin.iginx.util.TemplateUtil;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataPropertyInterpreter {

  private static final Logger LOGGER = LoggerFactory.getLogger(DataPropertyInterpreter.class);
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private final Map<String, NetworkService> networkMap = new HashMap<>();
  private static final String STATEMENT_PREFIX = ">data.property";
  private static final String DEFAULT_SQL = "SHOW COLUMNS;";

  private Session session;
  private String milvusHost;
  private int milvusPort;

  public DataPropertyInterpreter(
      Session session, String milvusHost, int milvusPort, String outfileDir) {
    this.milvusHost = milvusHost;
    this.milvusPort = milvusPort;
    this.session = session;
  }

  public boolean canInterpret(String sql, InterpreterContext context) {
    return sql.trim().startsWith(STATEMENT_PREFIX);
  }

  public InterpreterResult interpret(String statement, InterpreterContext context) {
    // 截取第一个空格之前的内容和之后的内容
    String[] strings = statement.trim().split(" ");
    String cmd = strings[0];
    String[] args = Arrays.copyOfRange(strings, 1, strings.length);

    try {
      switch (cmd) {
        case STATEMENT_PREFIX:
        case STATEMENT_PREFIX + ".network":
          return displayDataPropertyGraph(context, args, false);
        case STATEMENT_PREFIX + ".grouping":
        case STATEMENT_PREFIX + ".grouping.network":
          return displayDataPropertyGraph(context, args, true);
        case STATEMENT_PREFIX + ".tree":
          return displayDataPropertyTree(context, args);
        case STATEMENT_PREFIX + ".expand":
          return expandDataPropertyGraph(args);
        case STATEMENT_PREFIX + ".clear":
          return new InterpreterResult(
              InterpreterResult.Code.SUCCESS, InterpreterResult.Type.TEXT, "");
        default:
          throw new IllegalArgumentException("Invalid command: " + cmd);
      }
    } catch (Exception e) {
      return new InterpreterResult(InterpreterResult.Code.ERROR, ExceptionUtils.getStackTrace(e));
    }
  }

  private InterpreterResult displayDataPropertyGraph(
      InterpreterContext context, String[] args, boolean allowMerge) {
    List<String[]> paths = queryPathFromSql(String.join(" ", args));
    NetworkService networkService =
        new NetworkService(
            allowMerge, true, context.getParagraphId(), paths, session, milvusHost, milvusPort);
    networkMap.put(context.getParagraphId(), networkService);

    VelocityContext velocityContext = new VelocityContext();
    velocityContext.put("paragraphId", context.getParagraphId());

    String html = networkService.initNetwork(velocityContext);
    return new InterpreterResult(InterpreterResult.Code.SUCCESS, InterpreterResult.Type.HTML, html);
  }

  private InterpreterResult displayDataPropertyTree(InterpreterContext context, String[] args)
      throws JsonProcessingException {
    List<String[]> paths = queryPathFromSql(String.join(" ", args));
    String html = generateDataPropertyHtml(paths, context);
    return new InterpreterResult(InterpreterResult.Code.SUCCESS, InterpreterResult.Type.HTML, html);
  }

  public String generateDataPropertyHtml(List<String[]> paths, InterpreterContext context)
      throws JsonProcessingException {
    GraphData.Builder builder = new GraphData.Builder();
    for (String[] path : paths) {
      builder.addNode(path);
    }
    GraphData graphData = builder.build();
    String graphDataJson = MAPPER.writeValueAsString(graphData);

    VelocityContext velocityContext = new VelocityContext();
    velocityContext.put("paragraphId", context.getParagraphId());
    velocityContext.put("data", graphDataJson);
    return TemplateUtil.generate("templates/data-property-tree.vm", velocityContext);
  }

  private InterpreterResult expandDataPropertyGraph(String[] args) {
    Preconditions.checkArgument(args.length == 2, "Invalid number of arguments: " + args.length);
    String paragraphId = args[0];
    String nodeId = args[1];

    NetworkService networkService = networkMap.get(paragraphId);
    Preconditions.checkNotNull(networkService, "Network service not found: " + paragraphId);

    String msg = networkService.handleNodeClick(nodeId);
    return new InterpreterResult(InterpreterResult.Code.SUCCESS, InterpreterResult.Type.TEXT, msg);
  }

  private List<String[]> queryPathFromSql(String sql) {
    if (sql.isEmpty()) {
      return queryPathFromSql(DEFAULT_SQL);
    }

    SessionExecuteSqlResult sqlResult;
    try {
      sqlResult = session.executeSql(sql);
    } catch (SessionException e) {
      throw new RuntimeException("Failed to execute SQL: " + sql, e);
    }

    List<List<String>> queryList =
        sqlResult.getResultInList(false, FormatUtils.DEFAULT_TIME_FORMAT, null);
    int pathColumnIndex = queryList.get(0).indexOf("Path");
    if (pathColumnIndex == -1) {
      pathColumnIndex = queryList.get(0).indexOf("path");
    }
    if (pathColumnIndex == -1) {
      throw new IllegalArgumentException(
          "'Path' or 'path' column not found in the result: " + queryList.get(0));
    }
    List<String> paths = new ArrayList<>();
    for (int i = 1; i < queryList.size(); i++) {
      paths.add(queryList.get(i).get(pathColumnIndex));
    }
    Pattern pattern = Pattern.compile("\\.");
    return paths.stream().map(pattern::split).collect(Collectors.toList());
  }
}
