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
}
