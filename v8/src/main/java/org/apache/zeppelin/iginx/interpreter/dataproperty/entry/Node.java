package org.apache.zeppelin.iginx.interpreter.dataproperty.entry;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.google.common.base.Preconditions;
import java.util.*;

@JsonSerialize
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY)
public class Node {

  @JsonSerialize
  @JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY)
  public static class Style {}

  @JsonSerialize
  @JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY)
  public static class Data {
    private String label;

    public Data(String label) {
      this.label = Objects.requireNonNull(label);
    }
  }

  private List<String> children = new ArrayList<>();
  private String combo = null;
  private Data data;
  private Integer depth = null;
  private String id;
  private List<String> states = new ArrayList<>();
  private Style style = new Style();
  private String type = null;

  public Node(String id, String label) {
    this.id = Objects.requireNonNull(id);
    this.data = new Data(label);
  }

  public String getId() {
    return id;
  }

  public void addChildren(String id) {
    children.add(Objects.requireNonNull(id));
  }

  public void setDepth(int index) {
    Preconditions.checkArgument(index >= 0);
    depth = index;
  }
}
