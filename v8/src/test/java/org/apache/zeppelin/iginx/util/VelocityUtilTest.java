package org.apache.zeppelin.iginx.util;

import static org.junit.jupiter.api.Assertions.*;

import org.apache.velocity.VelocityContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class VelocityUtilTest {

  @ParameterizedTest
  @ValueSource(
      strings = {
        "templates/data-property-tree.vm",
      })
  void generate(String resourceName) {
    VelocityContext context = new VelocityContext();
    String result = VelocityUtil.generate(resourceName, context);
    System.out.println(result);
    assertNotNull(result);
  }
}
