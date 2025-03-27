package org.apache.zeppelin.iginx.interpreter.udfgenerator;

import cn.edu.tsinghua.iginx.session.Session;
import java.util.Arrays;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.zeppelin.iginx.interpreter.IginxDao;
import org.apache.zeppelin.interpreter.InterpreterResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UdfGeneratorInterpreter {
  private static final Logger LOGGER = LoggerFactory.getLogger(UdfGeneratorInterpreter.class);
  private static final String DEFAULT_GENERATE_UDF_FUNCTION = "generate_udf";
  private final IginxDao iginx;

  public UdfGeneratorInterpreter(Session session, String milvusHost, int milvusPort) {
    this.iginx = IginxDao.getInstance(session, milvusHost, milvusPort);
  }

  public boolean canInterpreter(String sql) {
    String cmd = sql.trim().split(" ")[0];
    switch (cmd) {
      case ">generate.udf.encode":
      case ">generate.udf.insert":
      case ">generate.udf.describe":
        return true;
      default:
        return false;
    }
  }

  public InterpreterResult interpret(String statement) {
    // 截取第一个空格之前的内容和之后的内容
    String[] strings = statement.trim().split(" ");
    String cmd = strings[0];
    String[] args = Arrays.copyOfRange(strings, 1, strings.length);

    try {
      switch (cmd) {
        case ">generate.udf.encode":
          return displayUdfGenerator(args, "Encode");
        case ">generate.udf.insert":
          return displayUdfGenerator(args, "Insert");
        case ">generate.udf.describe":
          return displayUdfGenerator(args, "Describe");
        default:
          throw new IllegalArgumentException("Invalid command: " + cmd);
      }
    } catch (Exception e) {
      return new InterpreterResult(InterpreterResult.Code.ERROR, ExceptionUtils.getStackTrace(e));
    }
  }

  private InterpreterResult displayUdfGenerator(String[] args, String type) {
    try {
      String description = String.join(" ", args);
      String udf = iginx.generateUdf(description, type, DEFAULT_GENERATE_UDF_FUNCTION).replace("```python", "").replace("```", "");
      LOGGER.info("displayUdfGenerator is {}", udf);
      return new InterpreterResult(
          InterpreterResult.Code.SUCCESS, InterpreterResult.Type.TEXT, udf);
    } catch (Exception e) {
      return new InterpreterResult(InterpreterResult.Code.ERROR, ExceptionUtils.getStackTrace(e));
    }
  }
}
