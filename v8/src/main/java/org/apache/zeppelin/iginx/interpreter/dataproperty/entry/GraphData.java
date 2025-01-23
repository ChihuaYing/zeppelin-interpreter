package org.apache.zeppelin.iginx.interpreter.dataproperty.entry;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.google.common.base.Preconditions;
import java.util.*;
import java.util.stream.Collectors;
import org.apache.commons.lang3.tuple.Pair;

@JsonSerialize
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY)
public class GraphData {
  private final List<Node> nodes = new ArrayList<>();
  private final List<EdgeData> edges = new ArrayList<>();
  private final List<ComboData> combos = new ArrayList<>();

  public static class Builder {
    private final Map<Pair<String, String>, EdgeData> edgeMap = new HashMap<>();
    private final Map<String, Node> nodeMap = new HashMap<>();
    private final String rootId = "rootId";
    private final String rootLabel = "Data Property";

    public Builder() {
      nodeMap.put(rootId, new Node(rootId, rootLabel));
    }

    public Builder addNode(String[] path) {
      addNode(path, 0);
      return this;
    }

    private Node addNode(String[] path, int index) {
      Preconditions.checkArgument(index >= 0);
      Preconditions.checkArgument(index <= path.length);
      Node node;
      if (index == 0) {
        node = nodeMap.get(rootId);
      } else {
        String id = Arrays.stream(path, 0, index).collect(Collectors.joining("."));
        node = nodeMap.computeIfAbsent(id, k -> new Node(id, path[index - 1]));
        node.setDepth(index);
      }
      if (index < path.length) {
        Node child = addNode(path, index + 1);
        node.addChildren(child.getId());
        edgeMap.computeIfAbsent(
            Pair.of(node.getId(), child.getId()), p -> new EdgeData(p.getKey(), p.getValue()));
      }
      return node;
    }

    public GraphData build() {
      GraphData graphData = new GraphData();
      graphData.nodes.addAll(nodeMap.values());
      graphData.edges.addAll(edgeMap.values());
      return graphData;
    }
  }
}
