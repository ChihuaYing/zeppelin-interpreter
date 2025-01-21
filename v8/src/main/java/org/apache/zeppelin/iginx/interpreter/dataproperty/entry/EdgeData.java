package org.apache.zeppelin.iginx.interpreter.dataproperty.entry;

import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import java.util.Objects;

@JsonSerialize
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY)
public class EdgeData {
  private String source;
  private String target;

  public EdgeData(String source, String target) {
    this.source = Objects.requireNonNull(source);
    this.target = Objects.requireNonNull(target);
  }
}
