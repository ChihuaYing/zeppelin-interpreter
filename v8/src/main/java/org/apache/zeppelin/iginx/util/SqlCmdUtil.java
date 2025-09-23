package org.apache.zeppelin.iginx.util;

public class SqlCmdUtil {
  public static String removeExtraSpaces(String str) {
    StringBuilder result = new StringBuilder();
    boolean lastCharWasSpace = false;
    for (char c : str.toCharArray()) {
      if (Character.isWhitespace(c)) {
        if (!lastCharWasSpace) {
          result.append(' ');
        }
        lastCharWasSpace = true;
      } else {
        result.append(c);
        lastCharWasSpace = false;
      }
    }
    return result.toString();
  }

  // SQL 状态机状态
  private enum State {
    DEFAULT, // 普通代码
    SINGLE_QUOTE, // 单引号字符串
    DOUBLE_QUOTE, // 双引号字符串
    BACK_QUOTE, // 反引号字符串
    LINE_COMMENT, // -- 单行注释
    BLOCK_COMMENT // /* */ 块注释
  }

  public static String removeComments(String sql) {
    if (sql == null) {
      return null;
    }

    StringBuilder result = new StringBuilder();
    State state = State.DEFAULT;
    int blockDepth = 0; // 块注释嵌套层数
    int length = sql.length();

    for (int i = 0; i < length; i++) {
      char c = sql.charAt(i);

      switch (state) {
        case DEFAULT:
          // 检测进入字符串
          if (c == '\'') {
            state = State.SINGLE_QUOTE;
            result.append(c);
          } else if (c == '"') {
            state = State.DOUBLE_QUOTE;
            result.append(c);
          } else if (c == '`') {
            state = State.BACK_QUOTE;
            result.append(c);
          }
          // 检测单行注释
          else if (c == '-' && i + 1 < length && sql.charAt(i + 1) == '-') {
            state = State.LINE_COMMENT;
            i++; // 跳过第二个 -
          }
          // 检测块注释
          else if (c == '/' && i + 1 < length && sql.charAt(i + 1) == '*') {
            state = State.BLOCK_COMMENT;
            blockDepth = 1;
            i++; // 跳过 *
          } else {
            result.append(c);
          }
          break;

        case SINGLE_QUOTE:
          result.append(c);
          if (c == '\'') {
            state = State.DEFAULT; // 字符串结束
          }
          break;

        case DOUBLE_QUOTE:
          result.append(c);
          if (c == '"') {
            state = State.DEFAULT; // 字符串结束
          }
          break;

        case BACK_QUOTE:
          result.append(c);
          if (c == '`') {
            state = State.DEFAULT; // 字符串结束
          }
          break;

        case LINE_COMMENT:
          // 跳过直到行尾
          if (c == '\n' || c == '\r') {
            state = State.DEFAULT;
            result.append(c); // 保留换行
          }
          break;

        case BLOCK_COMMENT:
          // 检测嵌套
          if (c == '/' && i + 1 < length && sql.charAt(i + 1) == '*') {
            blockDepth++;
            i++;
          } else if (c == '*' && i + 1 < length && sql.charAt(i + 1) == '/') {
            blockDepth--;
            i++;
            if (blockDepth == 0) {
              state = State.DEFAULT;
            }
          }
          break;
      }
    }

    return result.toString();
  }
}
