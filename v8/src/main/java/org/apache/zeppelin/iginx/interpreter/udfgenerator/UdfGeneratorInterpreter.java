package org.apache.zeppelin.iginx.interpreter.udfgenerator;

import cn.edu.tsinghua.iginx.session.Session;
import java.util.Arrays;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.velocity.VelocityContext;
import org.apache.zeppelin.iginx.interpreter.AbstractExtensionInterpreter;
import org.apache.zeppelin.iginx.util.VelocityUtil;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.apache.zeppelin.interpreter.InterpreterResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UdfGeneratorInterpreter extends AbstractExtensionInterpreter {
  private static final Logger LOGGER = LoggerFactory.getLogger(UdfGeneratorInterpreter.class);
  private static final String DEFAULT_GENERATE_UDF_FUNCTION = "generate_udf";

  public UdfGeneratorInterpreter(Session session, String milvusHost, int milvusPort) {
    super(session, milvusHost, milvusPort);
  }

  @Override
  public boolean canInterpret(String sql) {
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

  @Override
  public InterpreterResult interpret(String statement, InterpreterContext context) {
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
      String udf =
          iginx
              .generateUdf(description, type, DEFAULT_GENERATE_UDF_FUNCTION)
              .replace("```python", "")
              .replace("```", "")
              .trim();
      VelocityContext velocityContext = new VelocityContext();
      velocityContext.put("python_text", udf);
      String html = VelocityUtil.generate("templates/udf-generator.vm", velocityContext);

      return new InterpreterResult(
          InterpreterResult.Code.SUCCESS, InterpreterResult.Type.HTML, html);
    } catch (Exception e) {
      return new InterpreterResult(InterpreterResult.Code.ERROR, ExceptionUtils.getStackTrace(e));
    }
  }
}
