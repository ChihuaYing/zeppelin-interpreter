package org.apache.zeppelin.iginx.interpreter.dataproperty;

import cn.edu.tsinghua.iginx.exception.SessionException;
import cn.edu.tsinghua.iginx.session.Session;
import cn.edu.tsinghua.iginx.session.SessionExecuteSqlResult;
import cn.edu.tsinghua.iginx.utils.FormatUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.apache.commons.lang3.tuple.Pair;

public class IginxDao {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private final Session session;
  private final String milvusHost;
  private final int milvusPort;

  public IginxDao(Session session, String milvusHost, int milvusPort) {
    this.session = session;
    this.milvusHost = milvusHost;
    this.milvusPort = milvusPort;
  }

  public List<String[]> getPathOf(String sql) {
    if (sql.isEmpty()) {
      return getPathOf("SHOW COLUMNS;");
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

  public Multimap<String, String> getGroupingOf(Set<String> nodes) {
    // 使用 jackson 手动构建 JSON 字符串
    ArrayNode arrayNode = MAPPER.createArrayNode();
    for (String node : nodes) {
      arrayNode.add(node);
    }
    String nodesJson = arrayNode.toString();

    String sql =
        "select `merge(name)` as name, `merge(label)` as label from (select merge(*, str='"
            + nodesJson
            + "', host='"
            + milvusHost
            + "', port='"
            + milvusPort
            + "') from (show columns ###));";
    List<List<String>> queryList = getQueryList(sql);

    Multimap<String, String> groupingMap = HashMultimap.create();
    List<String> header = queryList.get(0);
    Preconditions.checkArgument(header.size() == 2, "Invalid header: " + header);
    Preconditions.checkArgument(header.get(0).equals("name"), "Invalid header: " + header);
    Preconditions.checkArgument(header.get(1).equals("label"), "Invalid header: " + header);

    for (int i = 1; i < queryList.size(); i++) {
      List<String> row = queryList.get(i);
      String name = row.get(0);
      String label = row.get(1);
      groupingMap.put(label, name);
    }

    return groupingMap;
  }

  private List<List<String>> getQueryList(String sql) {
    List<List<String>> queryList = null;
    try {
      SessionExecuteSqlResult sqlResult = session.executeSql(sql);
      queryList = sqlResult.getResultInList(false, FormatUtils.DEFAULT_TIME_FORMAT, "");
    } catch (Exception e) {
      throw new IllegalStateException("encounter error when executing sql statement", e);
    }
    if (queryList == null || queryList.size() <= 1) {
      throw new IllegalStateException("Invalid queryList or insufficient data, the sql is " + sql);
    }
    return queryList;
  }

  public Map<String, float[]> queryEmbeddingOfPaths(List<String> nodes) {
    Map<String, float[]> resultMap = new HashMap<>();
    if (nodes.isEmpty()) {
      return resultMap;
    }

    ArrayNode arrayNode = MAPPER.createArrayNode();
    for (String node : nodes) {
      arrayNode.add(node);
    }
    String nodesJson = arrayNode.toString();

    String sql =
        "select `get_embedding(path)` as path, `get_embedding(embedding)` as embedding from (select get_embedding(*, nodes='"
            + nodesJson
            + "', host='"
            + milvusHost
            + "', port='"
            + milvusPort
            + "') from (show columns ###));";

    SessionExecuteSqlResult sqlResult;
    try {
      sqlResult = session.executeSql(sql);
    } catch (SessionException e) {
      throw new RuntimeException("Failed to execute SQL: " + sql, e);
    }

    List<List<Object>> queryList = sqlResult.getValues();
    for (List<Object> row : queryList) {
      String path = new String((byte[]) row.get(0), StandardCharsets.UTF_8);
      byte[] embeddingBytes = (byte[]) row.get(1);
      FloatBuffer floatBuffer =
          ByteBuffer.wrap(embeddingBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
      float[] embedding = new float[floatBuffer.remaining()];
      floatBuffer.get(embedding);
      resultMap.put(path, embedding);
    }

    return resultMap;
  }

  public String askRelation(String nodeA, String nodeB) {
    String prompt = "你是一个概念大师，请你用短语给出“" + nodeA + "”和“" + nodeB + "”之间切实具体、简洁精炼的关系，最好不超过10个字。";
    return askBigModel(prompt);
  }

  public String askConcept(List<String> nodes) {
    String prompt =
        "你是一个概括大师，我将给你几个用‘.’分隔的中文短语，请你将它们概括成一个中文短语。注意仅需返回概括结果。\n需要概括的中文短语是: "
            + String.join(".", nodes);
    return askBigModel(prompt);
  }

  public String askBigModel(String prompt) {
    String sql = "select ask_big_model(*, prompt='" + prompt + "') from (show columns ###);";

    SessionExecuteSqlResult sqlResult;
    try {
      sqlResult = session.executeSql(sql);
    } catch (SessionException e) {
      throw new RuntimeException("Failed to execute SQL: " + sql, e);
    }

    List<List<Object>> queryList = sqlResult.getValues();
    for (List<Object> row : queryList) {
      return new String((byte[]) row.get(0), StandardCharsets.UTF_8);
    }
    throw new IllegalStateException("The result of ask_big_model is empty: " + sql);
  }

  public List<String> search(List<String> visiblePaths, String description) {
    ArrayNode arrayNode = MAPPER.createArrayNode();
    for (String path : visiblePaths) {
      arrayNode.add(path);
    }
    String pathsJson = arrayNode.toString();

    String sql =
        "select search_embedding(*, description='"
            + description
            + "', paths='"
            + pathsJson
            + "', host='"
            + milvusHost
            + "', port='"
            + milvusPort
            + "') from (show columns ###);";

    SessionExecuteSqlResult sqlResult;
    try {
      sqlResult = session.executeSql(sql);
    } catch (SessionException e) {
      throw new RuntimeException("Failed to execute SQL: " + sql, e);
    }

    List<String> paths = new ArrayList<>();
    for (List<Object> row : sqlResult.getValues()) {
      String pathSepDot = new String((byte[]) row.get(0), StandardCharsets.UTF_8);
      paths.add(pathSepDot);
    }
    return paths;
  }

  public Map<Pair<String, String>, Double> analyseMainRelation(
      List<String> sourcePaths, List<String> targetPaths) {
    ArrayNode sourceArrayNode = MAPPER.createArrayNode();
    for (String sourcePath : sourcePaths) {
      sourceArrayNode.add(sourcePath);
    }
    String sourcePathsJson = sourceArrayNode.toString();

    ArrayNode targetArrayNode = MAPPER.createArrayNode();
    for (String targetPath : targetPaths) {
      targetArrayNode.add(targetPath);
    }
    String targetPathsJson = targetArrayNode.toString();

    String sql =
        "select `analyse_relation(source)` as source, `analyse_relation(target)` as target, `analyse_relation(score)` as score"
            + " from (select analyse_relation(*, sources='"
            + sourcePathsJson
            + "', targets='"
            + targetPathsJson
            + "', host='"
            + milvusHost
            + "', port='"
            + milvusPort
            + "') from (show columns ###));";

    SessionExecuteSqlResult sqlResult;
    try {
      sqlResult = session.executeSql(sql);
    } catch (SessionException e) {
      throw new RuntimeException("Failed to execute SQL: " + sql, e);
    }

    Map<Pair<String, String>, Double> resultMap = new HashMap<>();
    List<List<Object>> queryList = sqlResult.getValues();
    for (List<Object> row : queryList) {
      String source = new String((byte[]) row.get(0), StandardCharsets.UTF_8);
      String target = new String((byte[]) row.get(1), StandardCharsets.UTF_8);
      double score = (double) row.get(2);
      resultMap.put(Pair.of(source, target), score);
    }
    return resultMap;
  }
}
