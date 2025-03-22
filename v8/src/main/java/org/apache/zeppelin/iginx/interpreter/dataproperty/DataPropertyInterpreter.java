package org.apache.zeppelin.iginx.interpreter.dataproperty;

import cn.edu.tsinghua.iginx.session.Session;
import cn.edu.tsinghua.iginx.utils.Pair;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Preconditions;
import java.util.*;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.velocity.VelocityContext;
import org.apache.zeppelin.iginx.interpreter.dataproperty.network.NetworkService;
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
    String cmd = sql.trim().split(" ")[0];
    switch (cmd) {
      case ">network":
      case ">network.grouping":
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
        case ">network":
          return displayDataPropertyGraph(context, args, false);
        case ">network.grouping":
          return displayDataPropertyGraph(context, args, true);
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
      InterpreterContext context, String[] args, boolean needMerge) {
    String pattern = String.join(" ", args);
    Pair<List<String[]>, String> pair = iginx.getPathOf(pattern);
    List<String[]> paths = pair.getK();
    String table = pair.getV();
    NetworkService networkService =
        new NetworkService(needMerge, true, context.getParagraphId(), paths, iginx, pattern);
    networkMap.put(context.getParagraphId(), networkService);

    VelocityContext velocityContext = new VelocityContext();
    velocityContext.put("paragraphId", context.getParagraphId());

    String html = networkService.initNetwork(velocityContext);

    InterpreterResult interpreterResult = new InterpreterResult(InterpreterResult.Code.SUCCESS);
    interpreterResult.add(InterpreterResult.Type.HTML, html);
    interpreterResult.add(InterpreterResult.Type.TABLE, table);
    return interpreterResult;
  }

  private InterpreterResult expandDataPropertyGraph(String[] args) {
    Preconditions.checkArgument(args.length >= 2, "Invalid number of arguments: " + args.length);
    String paragraphId = args[0];
    String nodeId = String.join(" ", Arrays.copyOfRange(args, 1, args.length));

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
