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
    interpreter = new DataPropertyInterpreter(null, null, 0, null);
  }

  @Test
  void testRenderDataProperty() throws IOException {
    List<String> paths = getTpchPaths();
    InterpreterContext context = getInterpreterContext();

    // Call the method to render data property
    String html = interpreter.generateDataPropertyHtml(paths, context);

    // 将结果写入临时文件并将路径打印出来
    Path path = Files.createTempFile("data_property", ".html");
    Files.write(path, html.getBytes());
    System.out.println("Data property html file path: " + path.toUri());
  }

  private List<String> getTpchPaths() {
    return Arrays.asList(
        "tpch.customer.c_acctbal",
        "tpch.customer.c_address",
        "tpch.customer.c_comment",
        "tpch.customer.c_custkey",
        "tpch.customer.c_mktsegment",
        "tpch.customer.c_name",
        "tpch.customer.c_nationkey",
        "tpch.customer.c_phone",
        "tpch.lineitem.l_comment",
        "tpch.lineitem.l_commitdate",
        "tpch.lineitem.l_discount",
        "tpch.lineitem.l_extendedprice",
        "tpch.lineitem.l_linenumber",
        "tpch.lineitem.l_linestatus",
        "tpch.lineitem.l_orderkey",
        "tpch.lineitem.l_partkey",
        "tpch.lineitem.l_quantity",
        "tpch.lineitem.l_receiptdate",
        "tpch.lineitem.l_returnflag",
        "tpch.lineitem.l_shipdate",
        "tpch.lineitem.l_shipinstruct",
        "tpch.lineitem.l_shipmode",
        "tpch.lineitem.l_suppkey",
        "tpch.lineitem.l_tax",
        "tpch.nation.n_comment",
        "tpch.nation.n_name",
        "tpch.nation.n_nationkey",
        "tpch.nation.n_regionkey",
        "tpch.orders.o_clerk",
        "tpch.orders.o_comment",
        "tpch.orders.o_custkey",
        "tpch.orders.o_orderdate",
        "tpch.orders.o_orderkey",
        "tpch.orders.o_orderpriority",
        "tpch.orders.o_orderstatus",
        "tpch.orders.o_shippriority",
        "tpch.orders.o_totalprice",
        "tpch.part.p_brand",
        "tpch.part.p_comment",
        "tpch.part.p_container",
        "tpch.part.p_mfgr",
        "tpch.part.p_name",
        "tpch.part.p_partkey",
        "tpch.part.p_retailprice",
        "tpch.part.p_size",
        "tpch.part.p_type",
        "tpch.partsupp.ps_availqty",
        "tpch.partsupp.ps_comment",
        "tpch.partsupp.ps_partkey",
        "tpch.partsupp.ps_suppkey",
        "tpch.partsupp.ps_supplycost",
        "tpch.region.r_comment",
        "tpch.region.r_name",
        "tpch.region.r_regionkey",
        "tpch.supplier.s_acctbal",
        "tpch.supplier.s_address",
        "tpch.supplier.s_comment",
        "tpch.supplier.s_name",
        "tpch.supplier.s_nationkey",
        "tpch.supplier.s_phone",
        "tpch.supplier.s_suppkey");
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
