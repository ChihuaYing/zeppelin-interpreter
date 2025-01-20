package org.apache.zeppelin.iginx.util;

import static org.junit.jupiter.api.Assertions.*;

import org.apache.velocity.VelocityContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class TemplateUtilTest {

  @ParameterizedTest
  @ValueSource(
      strings = {
        "templates/data-property-tree.vm",
      })
  void generate(String resourceName) {
    VelocityContext context = new VelocityContext();
    String result = TemplateUtil.generate(resourceName, context);
    System.out.println(result);
    assertNotNull(result);
  }
}
