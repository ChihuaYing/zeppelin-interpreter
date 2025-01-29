package org.apache.zeppelin.iginx.interpreter.dataproperty.entry;

import lombok.Value;

@Value
public class Relation {
  String from;
  String to;
  Double score;
  String relation;

  public Relation(String from, String to, Double score, String relation) {
    this.from = from;
    this.to = to;
    this.score = score;
    this.relation = relation.replace("[^a-zA-Z0-9]", " ");
  }
}
