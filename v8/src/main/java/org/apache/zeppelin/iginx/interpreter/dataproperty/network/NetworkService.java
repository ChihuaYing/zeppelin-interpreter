package org.apache.zeppelin.iginx.interpreter.dataproperty.network;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.filter.PropertyFilter;
import com.google.common.collect.Multimap;
import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.velocity.VelocityContext;
import org.apache.zeppelin.iginx.interpreter.dataproperty.IginxDao;
import org.apache.zeppelin.iginx.util.VelocityUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NetworkService {
  private static final Logger LOGGER = LoggerFactory.getLogger(NetworkService.class);
  private static final Double RELATION_THRESHOLD = 0.8; // 关系阈值
  private static final Integer RELATION_DEPTH_LEVEL = 3; // 关系深度层级
  private static final Integer MERGE_MIN_SIZE = 5; // 需要聚类的最小值

  private Boolean needMerge; // 是否需要合并
  private Boolean needRelation; // 是否需要计算关系
  private String paragraphId;
  private List<String[]> columnPath;
  private NetworkTreeNode root;
  private IginxDao iginx;
  private Map<String, Map<String, Relation>> relationMap = new ConcurrentHashMap<>();

  public NetworkService(
      Boolean needMerge,
      Boolean needRelation,
      String paragraphId,
      List<String[]> columnPath,
      IginxDao iginx) {
    this.needMerge = needMerge;
    this.needRelation = needRelation;
    this.paragraphId = paragraphId;
    this.columnPath = columnPath;
    this.iginx = iginx;
  }

  public String initNetwork(VelocityContext velocityContext) {
    LOGGER.info("initNetwork: {} {} {}", needMerge, needRelation, paragraphId);
    root = new NetworkTreeNode("rootId", "数据资产", 0);
    buildForest(root, columnPath);
    if (needMerge) {
      LOGGER.info("before merge, the size is：{}", root.getChildren().size());
      mergeForest(root);
      LOGGER.info("after merge, the size is：{}", root.getChildren().size());
    }
    List<NetworkTreeNode> nodeList = new ArrayList<>();
    nodeList.add(root);
    root.setExpanded(true);
    root.setShown(true);
    for (NetworkTreeNode childNode : root.getChildren().values()) {
      childNode.setShown(true);
      nodeList.add(childNode);
    }
    PropertyFilter filter =
        (Object object, String name, Object value) -> {
          return "id".equals(name) || "name".equals(name) || "depth".equals(name);
        };
    String nodeString = JSON.toJSONString(nodeList, filter);
    LOGGER.info("the nodeString is {}", nodeString);

    String relationString = "";
    if (needRelation) {
      //      addEmbedding(root);
      List<Relation> relationList = calculateNodeRelation(root);
      relationString = JSON.toJSONString(relationList);
      LOGGER.info("the relationString is {}", relationString);
    }
    if (relationString.isEmpty()) {
      LOGGER.info("relationString is empty");
      relationString = "[]";
    }

    velocityContext.put("nodeList", nodeString);
    velocityContext.put("relationList", relationString);
    return VelocityUtil.generate("templates/data-property.vm", velocityContext);
  }

  public String handleNodeClick(String nodeId) {
    long startTime = System.currentTimeMillis();
    LOGGER.info("handleNodeClick");
    NetworkTreeNode node = getNodeById(nodeId);
    if (node == null) {
      LOGGER.error("Node not found for id: {}", nodeId);
      return "{}";
    }

    JSONObject result = new JSONObject();
    JSONObject addMap = new JSONObject();
    JSONObject removeMap = new JSONObject();

    if (node.getExpanded()) {
      throw new IllegalStateException("Node is already expanded");
      //      collapseNode(node, addMap, removeMap);
      //      node.setExpanded(false);
    } else {
      expandNode(node, addMap);
      node.setExpanded(true);
    }

    result.put("add", addMap);
    result.put("remove", removeMap);
    long endTime = System.currentTimeMillis();
    LOGGER.info("handleNodeClick run time：" + (endTime - startTime) + "ms");
    return result.toString();
  }

  // todo:数据量很大时，考虑先只build前几层？
  private void buildForest(NetworkTreeNode root, List<String[]> columnPath) {
    long startTime = System.currentTimeMillis();
    // 使用并行流处理 columnPath
    columnPath
        .parallelStream()
        .forEach(
            path -> {
              NetworkTreeNode currentNode = root;

              for (String nodeName : path) {
                // 使用同步块来保证线程安全
                synchronized (currentNode) {
                  NetworkTreeNode childNode = currentNode.getChildren().get(nodeName);
                  if (childNode == null) {
                    childNode =
                        new NetworkTreeNode(
                            currentNode.getId() + "." + nodeName, // 生成节点ID
                            nodeName, // 节点名称
                            currentNode.getDepth() + 1 // 父节点深度 + 1
                            );
                    currentNode.getChildren().put(nodeName, childNode);
                  }
                  currentNode = childNode;
                }
              }
            });

    long endTime = System.currentTimeMillis();
    LOGGER.info("buildForest run time：" + (endTime - startTime) + "ms");
  }

  // todo:数据量很大时，updateNodes会几乎遍历所有结点，比较耗时，后续考虑借鉴懒标记思想优化？
  private void mergeForest(NetworkTreeNode root) {
    Set<String> nodesSet = new HashSet<>();
    for (NetworkTreeNode childNode : root.getChildren().values()) {
      nodesSet.add(childNode.getName());
    }

    if (nodesSet.size() < MERGE_MIN_SIZE) {
      LOGGER.info("the size of the forest is too small");
      return;
    }

    Multimap<String, String> groupingMap = iginx.getGroupingOf(nodesSet);

    Map<String, List<NetworkTreeNode>> labelToNodesMap = new HashMap<>();
    for (Map.Entry<String, Collection<String>> group : groupingMap.asMap().entrySet()) {
      List<NetworkTreeNode> nodesToMerge = new ArrayList<>();
      for (String nodeName : group.getValue()) {
        NetworkTreeNode node = root.getChildren().get(nodeName);
        if (node == null) {
          throw new IllegalStateException("Node not found for name: " + nodeName);
        }
        nodesToMerge.add(node);
      }
      labelToNodesMap.put(group.getKey(), nodesToMerge);
    }

    // 对每个label进行合并
    labelToNodesMap
        .entrySet()
        .parallelStream()
        .forEach(
            entry -> {
              List<NetworkTreeNode> nodesToMerge = entry.getValue();
              if (nodesToMerge.size() > 1) {
                String mergedName =
                    iginx.askConcept(
                        nodesToMerge.stream()
                            .map(NetworkTreeNode::getName)
                            .collect(Collectors.toList()));
                NetworkTreeNode mergedNode =
                    new NetworkTreeNode("rootId." + mergedName, mergedName, 1);
                nodesToMerge.forEach(
                    node -> {
                      mergedNode.getChildren().put(node.getName(), node);
                      updateNodes(node, mergedNode.getName());
                      synchronized (root) {
                        root.getChildren().remove(node.getName());
                      }
                    });
                synchronized (root) {
                  root.getChildren().put(mergedNode.getName(), mergedNode);
                }
              }
            });
  }

  private void updateNodes(NetworkTreeNode node, String mergeRoot) {
    node.setMergedRoot(mergeRoot);
    for (NetworkTreeNode childNode : node.getChildren().values()) {
      updateNodes(childNode, mergeRoot);
    }
  }

  private String loadHtmlTemplate() {
    String htmlTemplate = "static/vis/network.html";
    try (InputStream inputStream =
        this.getClass().getClassLoader().getResourceAsStream(htmlTemplate)) {
      BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
      StringBuilder content = new StringBuilder();
      String line;
      while ((line = reader.readLine()) != null) {
        content.append(line).append("\n");
      }
      return content.toString();
    } catch (IOException e) {
      LOGGER.warn("load show columns to network error", e);
    }
    return htmlTemplate;
  }

  private NetworkTreeNode getNodeById(String nodeId) {
    String[] ids = nodeId.split("\\.");
    NetworkTreeNode currentNode = root;
    for (int i = 1; i < ids.length; i++) {
      currentNode = currentNode.getChildren().get(ids[i]);
      if (currentNode == null) return null;
    }
    return currentNode;
  }

  private void expandNode(NetworkTreeNode node, JSONObject addMap) {
    JSONArray nodes = new JSONArray();
    JSONArray edges = new JSONArray();
    JSONArray links = new JSONArray();
    if (needRelation) {
      //      addEmbedding(node);
      List<Relation> addRelations = calculateNodeRelation(node);
      links = getRelationLinks(addRelations);
    }

    for (NetworkTreeNode child : node.getChildren().values()) {
      child.setShown(true);
      JSONObject nodeData = new JSONObject();
      nodeData.put("id", child.getNetworkId());
      nodeData.put("name", child.getName());
      nodeData.put("depth", child.getDepth());
      nodes.add(nodeData);

      JSONObject edgeData = new JSONObject();
      edgeData.put("from", node.getNetworkId());
      edgeData.put("to", child.getNetworkId());
      edges.add(edgeData);
    }

    addMap.put("nodes", nodes);
    addMap.put("edges", edges);
    addMap.put("links", links);
  }

  //  private void collapseNode(NetworkTreeNode node, JSONObject addMap, JSONObject removeMap) {
  //    JSONArray nodes = new JSONArray();
  //    for (NetworkTreeNode child : node.getChildren().values()) {
  //      collectShownNodes(child, nodes);
  //    }
  //    removeMap.put("nodes", nodes);
  //
  //    if (needRelation) {
  //      List<Relation> addRelations = analyseNodeRelation();
  //      JSONArray links = getRelationLinks(addRelations);
  //      addMap.put("links", links);
  //    }
  //  }

  private JSONArray getRelationLinks(List<Relation> addRelations) {
    JSONArray links = new JSONArray();
    for (Relation relation : addRelations) {
      JSONObject relationJson = new JSONObject();
      relationJson.put("from", relation.getFrom());
      relationJson.put("to", relation.getTo());
      relationJson.put("relation", relation.getRelation());
      links.add(relationJson);
    }
    return links;
  }

  private void collectShownNodes(NetworkTreeNode node, JSONArray nodes) {
    if (node.getShown()) {
      JSONObject nodeData = new JSONObject();
      nodeData.put("id", node.getNetworkId());
      nodes.add(nodeData);
      node.setShown(false);
      for (NetworkTreeNode child : node.getChildren().values()) {
        collectShownNodes(child, nodes); // 递归处理
      }
    }
  }

  private void addEmbedding(NetworkTreeNode node) {
    LOGGER.info("begin addEmbedding for {}'s children", node.getId());
    List<String> paths = new ArrayList<>();
    for (NetworkTreeNode childNode : node.getChildren().values()) {
      paths.add(childNode.getEmbeddingId());
    }
    Map<String, float[]> embeddings = iginx.queryEmbeddingOfPaths(paths);
    for (NetworkTreeNode childNode : node.getChildren().values()) {
      childNode.setEmbedding(embeddings.get(childNode.getEmbeddingId()));
    }
  }

  private List<Relation> calculateNodeRelation(NetworkTreeNode node) {
    LOGGER.info("calculateNodeRelation: nodeId is {}", node.getId());
    if (node.getDepth() >= RELATION_DEPTH_LEVEL) return new ArrayList<>();
    List<NetworkTreeNode> visibleNodes = getVisibleNodes();
    LOGGER.info("the size of visibleNodes is {}", visibleNodes.size());

    List<String> sourcePaths = new ArrayList<>();
    Map<String, String> sourceEmbeddingId2NetworkId = new HashMap<>();
    for (NetworkTreeNode childNode : node.getChildren().values()) {
      sourcePaths.add(childNode.getEmbeddingId());
      sourceEmbeddingId2NetworkId.put(childNode.getEmbeddingId(), childNode.getNetworkId());
    }

    List<String> targetPaths = new ArrayList<>();
    Map<String, String> targetEmbeddingId2NetworkId = new HashMap<>();
    for (NetworkTreeNode visibleNode : visibleNodes) {
      targetPaths.add(visibleNode.getEmbeddingId());
      targetEmbeddingId2NetworkId.put(visibleNode.getEmbeddingId(), visibleNode.getNetworkId());
    }

    List<Relation> addRelations = new ArrayList<>();
    Map<Pair<String, String>, Double> relationMaps =
        iginx.analyseMainRelation(sourcePaths, targetPaths);
    for (Map.Entry<Pair<String, String>, Double> entry : relationMaps.entrySet()) {
      String sourceEmbeddingId = entry.getKey().getLeft();
      String targetEmbeddingId = entry.getKey().getRight();
      double score = entry.getValue();

      String from = sourceEmbeddingId2NetworkId.get(sourceEmbeddingId);
      String to = targetEmbeddingId2NetworkId.get(targetEmbeddingId);

      if (from == null || to == null) {
        throw new IllegalStateException(
            "Node not found for embeddingId: " + sourceEmbeddingId + " or " + targetEmbeddingId);
      }

      Relation relation = new Relation(from, to, score);
      addRelations.add(relation);
    }

    LOGGER.info("analyseNodeRelation");
    addRelations
        .parallelStream()
        .forEach(
            relation -> {
              String[] parts1 = relation.getFrom().split("\\.");
              String[] parts2 = relation.getTo().split("\\.");
              String name1 = parts1[parts1.length - 1];
              String name2 = parts2[parts2.length - 1];
              String relationMsg = iginx.askRelation(name1, name2);
              relation.setRelation(relationMsg);
            });

    return addRelations;
  }

  private List<NetworkTreeNode> getVisibleNodes() {
    List<NetworkTreeNode> result = new ArrayList<>();
    for (NetworkTreeNode node : root.getChildren().values()) {
      getVisibleNodes(node, result);
    }
    return result;
  }

  private void getVisibleNodes(NetworkTreeNode node, List<NetworkTreeNode> result) {
    if (node.getShown()) {
      result.add(node);
      for (NetworkTreeNode childNode : node.getChildren().values()) {
        getVisibleNodes(childNode, result);
      }
    }
  }
}
