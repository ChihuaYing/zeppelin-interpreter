package org.apache.zeppelin.iginx.interpreter.dataproperty.entry;

import lombok.Value;

@Value
public class Relation {
  String from;
  String to;
  Double score;
  String relation;
}
