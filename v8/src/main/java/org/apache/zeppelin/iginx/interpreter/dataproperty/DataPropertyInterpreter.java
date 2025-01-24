package org.apache.zeppelin.iginx.interpreter.dataproperty;

import cn.edu.tsinghua.iginx.session.Session;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Preconditions;
import java.util.*;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.velocity.VelocityContext;
import org.apache.zeppelin.iginx.interpreter.dataproperty.entry.GraphData;
import org.apache.zeppelin.iginx.interpreter.dataproperty.network.NetworkService;
import org.apache.zeppelin.iginx.util.VelocityUtil;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DataPropertyInterpreter {

  private static final Logger LOGGER = LoggerFactory.getLogger(DataPropertyInterpreter.class);
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private final Map<String, NetworkService> networkMap = new HashMap<>();
  private static final String INTERNAL_STATEMENT_PREFIX = ">data.property";

  private final IginxDao iginx;

  public DataPropertyInterpreter(Session session, String milvusHost, int milvusPort) {
    this.iginx = new IginxDao(session, milvusHost, milvusPort);
  }

  public boolean canInterpret(String sql, InterpreterContext context) {
    switch (sql.trim()) {
      case ">network.asset.data":
      case ">network.asset.data.grouping":
      case ">network":
      case ">network.grouping":
      case ">tree":
      case INTERNAL_STATEMENT_PREFIX + ".expand":
      case INTERNAL_STATEMENT_PREFIX + ".search":
      case INTERNAL_STATEMENT_PREFIX + ".clear":
        return true;
      default:
        return false;
    }
  }

  public InterpreterResult interpret(String statement, InterpreterContext context) {
    // 截取第一个空格之前的内容和之后的内容
    String[] strings = statement.trim().split(" ");
    String cmd = strings[0];
    String[] args = Arrays.copyOfRange(strings, 1, strings.length);

    try {
      switch (cmd) {
        case ">network.asset.data":
        case ">network":
          return displayDataPropertyGraph(context, args, false);
        case "network.asset.data.grouping":
        case "network.grouping":
          return displayDataPropertyGraph(context, args, true);
        case ">tree":
          return displayDataPropertyTree(context, args);
        case INTERNAL_STATEMENT_PREFIX + ".expand":
          return expandDataPropertyGraph(args);
        case INTERNAL_STATEMENT_PREFIX + ".search":
          return searchDataProperty(args);
        case INTERNAL_STATEMENT_PREFIX + ".clear":
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
    List<String[]> paths = iginx.getPathOf(String.join(" ", args));
    NetworkService networkService =
        new NetworkService(allowMerge, true, context.getParagraphId(), paths, iginx);
    networkMap.put(context.getParagraphId(), networkService);

    VelocityContext velocityContext = new VelocityContext();
    velocityContext.put("paragraphId", context.getParagraphId());

    String html = networkService.initNetwork(velocityContext);
    return new InterpreterResult(InterpreterResult.Code.SUCCESS, InterpreterResult.Type.HTML, html);
  }

  private InterpreterResult displayDataPropertyTree(InterpreterContext context, String[] args)
      throws JsonProcessingException {
    List<String[]> paths = iginx.getPathOf(String.join(" ", args));
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
    return VelocityUtil.generate("templates/data-property-tree.vm", velocityContext);
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

  private InterpreterResult searchDataProperty(String[] args) {
    if (args.length < 2) {
      return new InterpreterResult(InterpreterResult.Code.ERROR, "Empty search description");
    }
    String paragraphId = args[0];
    String description = String.join(" ", Arrays.copyOfRange(args, 1, args.length));

    NetworkService networkService = networkMap.get(paragraphId);
    Preconditions.checkNotNull(networkService, "Network service not found: " + paragraphId);

    String msg = networkService.handleSearch(description);
    return new InterpreterResult(InterpreterResult.Code.SUCCESS, InterpreterResult.Type.TEXT, msg);
  }
}
