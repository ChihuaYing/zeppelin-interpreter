package cn.edu.tsinghua;

import cn.edu.tsinghua.iginx.exception.SessionException;
import cn.edu.tsinghua.iginx.pool.SessionPool;
import cn.edu.tsinghua.iginx.session.Session;
import cn.edu.tsinghua.iginx.thrift.DataType;
import com.google.common.collect.Lists;

import java.io.*;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;

public class SchemaPileDataset {

  public static final String FILENAME = "schemapile-perm.json.gz";
  public static final String IGINX_HOST = "127.0.0.1";
  public static final int IGINX_PORT = 6888;
  public static final int BATCH_SIZE = 1000;

  protected final List<IginxColumn> data;

  interface SessionFactory {
    Session createSession() throws SessionException;
  }

  public SchemaPileDataset() throws IOException {
    String json = readGzipFileFromResources(FILENAME);
    SchemaPile schemaPile = new SchemaPile(json);
    this.data = schemaPile.toIginxColumn();
  }

  public static void main(String[] args) throws Exception {
    SessionFactory sessionFactory = () -> {
      Session session = new Session(IGINX_HOST, IGINX_PORT);
      session.openSession();
      return session;
    };

    System.out.println("Preparing to insert data...");
    SchemaPileDataset dataset = new SchemaPileDataset();
    dataset.insertInto(sessionFactory);
  }

  private static String readGzipFileFromResources(String filename) throws IOException {
    ClassLoader classLoader = SchemaPileDataset.class.getClassLoader();
    try (InputStream resourceStream = classLoader.getResourceAsStream(filename);
         GZIPInputStream gzip = new GZIPInputStream(resourceStream);
         InputStreamReader reader = new InputStreamReader(gzip);
         BufferedReader in = new BufferedReader(reader)) {
      StringBuilder sb = new StringBuilder();
      String line;
      while ((line = in.readLine()) != null) {
        sb.append(line);
      }
      return sb.toString();
    }
  }

  private void insertInto(SessionFactory sessionFactory) {
    List<List<IginxColumn>> partitions = Lists.partition(data, BATCH_SIZE);
    try (ProgressBar progressBar =
             new ProgressBarBuilder()
                 .setTaskName("Inserting Data")
                 .setInitialMax(partitions.size())
                 .build()) {
      partitions
          .parallelStream()
          .forEach(
              partition -> {
                try {
                  insertDataInto(sessionFactory, partition);
                  progressBar.step();
                } catch (SessionException e) {
                  throw new RuntimeException(e);
                }
              });
    }
  }

  private static void insertDataInto(SessionFactory sessionFactory, List<IginxColumn> data)
      throws SessionException {

    int maxLen = data.stream().mapToInt(column -> column.getValues().length).max().orElse(0);
    long[] keys = IntStream.range(0, maxLen).mapToLong(i -> i).toArray();
    List<String> paths = new ArrayList<>();
    List<DataType> types = new ArrayList<>();
    List<Object[]> values = new ArrayList<>();

    for (IginxColumn column : data) {
      paths.add(column.getPath());
      types.add(column.getType());
      values.add(Arrays.copyOf(column.getValues(), maxLen));
    }

    Session session = sessionFactory.createSession();
    try {
      session.insertColumnRecords(paths, keys, values.toArray(), types);
    } finally {
      session.closeSession();
    }
  }
}
