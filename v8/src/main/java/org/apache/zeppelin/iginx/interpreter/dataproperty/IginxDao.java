package org.apache.zeppelin.iginx.interpreter.dataproperty;

import cn.edu.tsinghua.iginx.exception.SessionException;
import cn.edu.tsinghua.iginx.session.Column;
import cn.edu.tsinghua.iginx.session.Session;
import cn.edu.tsinghua.iginx.session.SessionExecuteSqlResult;
import cn.edu.tsinghua.iginx.thrift.DataType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.google.common.base.Preconditions;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.lang3.StringUtils;
import org.apache.zeppelin.iginx.interpreter.dataproperty.entry.ClusterNode;
import org.apache.zeppelin.iginx.interpreter.dataproperty.entry.Relation;
import org.apache.zeppelin.iginx.interpreter.dataproperty.entry.SearchedNode;

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

  public List<Column> getPathOf(String iginxPattern) {
    Preconditions.checkArgument(StringUtils.isNotBlank(iginxPattern));

    String sql = String.format("select path, type from (show columns %s);", iginxPattern);
    List<List<Object>> values = executeSql(sql);

    List<Column> columns = new ArrayList<>();
    for (List<Object> row : values) {
      String path = new String((byte[]) row.get(0), StandardCharsets.UTF_8);
      String typeStr = new String((byte[]) row.get(1), StandardCharsets.UTF_8);
      DataType type = DataType.valueOf(typeStr);
      columns.add(new Column(path, type));
    }

    return columns;
  }

  public Multimap<ClusterNode, String> getGroupingOf(String iginxPattern, String function) {
    Preconditions.checkArgument(StringUtils.isNotBlank(iginxPattern));
    Preconditions.checkArgument(StringUtils.isNotBlank(function));

    String sql =
        String.format(
            "select `%s(path)`, `%<s(cluster)`"
                + " from (select %<s(*, pattern='%s', host='%s', port='%d') from (show columns ###));",
            function, iginxPattern, milvusHost, milvusPort);

    List<List<Object>> values = executeSql(sql);
    Multimap<ClusterNode, String> groupingMap = HashMultimap.create();
    for (List<Object> row : values) {
      String path = new String((byte[]) row.get(0), StandardCharsets.UTF_8);
      String clusterKey = new String((byte[]) row.get(1), StandardCharsets.UTF_8);
      groupingMap.put(new ClusterNode(clusterKey, "<description example>: " + clusterKey), path);
    }

    return groupingMap;
  }

  public List<SearchedNode> search(
      String iginxPattern, String description, String topK, String function) {
    // todo: 把 topK 放到 UDF 中
    // todo: 目前 score 值是直接放进去的，后续改为从 UDF 中获得
    Preconditions.checkArgument(StringUtils.isNotBlank(iginxPattern));
    Preconditions.checkArgument(StringUtils.isNotBlank(topK));
    Preconditions.checkArgument(StringUtils.isNotBlank(description));
    Preconditions.checkArgument(StringUtils.isNotBlank(function));

    String sql =
        String.format(
            "select `%s(path)`"
                + " from (select %<s(*, description='%s', pattern='%s', host='%s', port='%d') from (show columns ###));",
            function, description, iginxPattern, milvusHost, milvusPort);

    List<List<Object>> values = executeSql(sql);

    List<SearchedNode> pairs = new ArrayList<>();
    for (List<Object> row : values) {
      String pathSepDot = new String((byte[]) row.get(0), StandardCharsets.UTF_8);
      pairs.add(new SearchedNode(pathSepDot, 1.0));
    }
    return pairs;
  }

  public List<Relation> analyseMainRelation(
      List<String> sourcePaths, List<String> targetPaths, String function) {
    Preconditions.checkNotNull(sourcePaths);
    Preconditions.checkNotNull(targetPaths);
    Preconditions.checkArgument(StringUtils.isNotBlank(function));

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
        String.format(
            "select `%s(source)`, `%<s(target)`, `%<s(score)`, `%<s(description)`"
                + " from (select %<s(*, sources='%s', targets='%s', host='%s', port='%d') from (show columns ###));",
            function, sourcePathsJson, targetPathsJson, milvusHost, milvusPort);

    List<List<Object>> values = executeSql(sql);

    List<Relation> relations = new ArrayList<>();
    for (List<Object> row : values) {
      String source = new String((byte[]) row.get(0), StandardCharsets.UTF_8);
      String target = new String((byte[]) row.get(1), StandardCharsets.UTF_8);
      double score = (double) row.get(2);
      String description = new String((byte[]) row.get(3), StandardCharsets.UTF_8);
      relations.add(new Relation(source, target, score, description));
    }
    return relations;
  }

  private List<List<Object>> executeSql(String sql) {
    SessionExecuteSqlResult sqlResult;
    try {
      sqlResult = session.executeSql(sql);
    } catch (SessionException e) {
      throw new RuntimeException("Failed to execute SQL: " + sql, e);
    }
    return sqlResult.getValues();
  }
}
