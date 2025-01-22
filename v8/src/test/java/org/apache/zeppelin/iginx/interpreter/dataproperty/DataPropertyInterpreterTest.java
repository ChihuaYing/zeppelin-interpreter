package org.apache.zeppelin.iginx.interpreter.dataproperty;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.apache.zeppelin.interpreter.InterpreterContext;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class DataPropertyInterpreterTest {
  private DataPropertyInterpreter interpreter;

  @BeforeEach
  void setUp() {
    interpreter = new DataPropertyInterpreter(null, null, 0);
  }

  @Test
  void testRenderDataProperty() throws IOException {
    List<String[]> paths = getTpchPaths();
    InterpreterContext context = getInterpreterContext();

    // Call the method to render data property
    String html = interpreter.generateDataPropertyHtml(paths, context);

    // 将结果写入临时文件并将路径打印出来
    Path path = Files.createTempFile("data_property", ".html");
    Files.write(path, html.getBytes());
    System.out.println("Data property html file path: " + path.toUri());
  }

  private List<String[]> getTpchPaths() {
    return Arrays.asList(
        new String[] {"customer", "c_acctbal"},
        new String[] {"customer", "c_address"},
        new String[] {"customer", "c_comment"},
        new String[] {"customer", "c_custkey"},
        new String[] {"customer", "c_mktsegment"},
        new String[] {"customer", "c_name"},
        new String[] {"customer", "c_nationkey"},
        new String[] {"customer", "c_phone"},
        new String[] {"lineitem", "l_comment"},
        new String[] {"lineitem", "l_commitdate"},
        new String[] {"lineitem", "l_discount"},
        new String[] {"lineitem", "l_extendedprice"},
        new String[] {"lineitem", "l_linenumber"},
        new String[] {"lineitem", "l_linestatus"},
        new String[] {"lineitem", "l_orderkey"},
        new String[] {"lineitem", "l_partkey"},
        new String[] {"lineitem", "l_quantity"},
        new String[] {"lineitem", "l_receiptdate"},
        new String[] {"lineitem", "l_returnflag"},
        new String[] {"lineitem", "l_shipdate"},
        new String[] {"lineitem", "l_shipinstruct"},
        new String[] {"lineitem", "l_shipmode"},
        new String[] {"lineitem", "l_suppkey"},
        new String[] {"lineitem", "l_tax"},
        new String[] {"nation", "n_comment"},
        new String[] {"nation", "n_name"},
        new String[] {"nation", "n_nationkey"},
        new String[] {"nation", "n_regionkey"},
        new String[] {"orders", "o_clerk"},
        new String[] {"orders", "o_comment"},
        new String[] {"orders", "o_custkey"},
        new String[] {"orders", "o_orderdate"},
        new String[] {"orders", "o_orderkey"},
        new String[] {"orders", "o_orderpriority"},
        new String[] {"orders", "o_orderstatus"},
        new String[] {"orders", "o_shippriority"},
        new String[] {"orders", "o_totalprice"},
        new String[] {"part", "p_brand"},
        new String[] {"part", "p_comment"},
        new String[] {"part", "p_container"},
        new String[] {"part", "p_mfgr"},
        new String[] {"part", "p_name"},
        new String[] {"part", "p_partkey"},
        new String[] {"part", "p_retailprice"},
        new String[] {"part", "p_size"},
        new String[] {"part", "p_type"},
        new String[] {"partsupp", "ps_availqty"},
        new String[] {"partsupp", "ps_comment"},
        new String[] {"partsupp", "ps_partkey"},
        new String[] {"partsupp", "ps_suppkey"},
        new String[] {"partsupp", "ps_supplycost"},
        new String[] {"region", "r_comment"},
        new String[] {"region", "r_name"},
        new String[] {"region", "r_regionkey"},
        new String[] {"supplier", "s_acctbal"},
        new String[] {"supplier", "s_address"},
        new String[] {"supplier", "s_comment"},
        new String[] {"supplier", "s_name"},
        new String[] {"supplier", "s_nationkey"},
        new String[] {"supplier", "s_phone"},
        new String[] {"supplier", "s_suppkey"});
  }

  private InterpreterContext getInterpreterContext() {
    return new InterpreterContext(
        "noteId",
        "paragraphId",
        "replName",
        "paragraphTitle",
        "paragraphText",
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null);
  }
}
