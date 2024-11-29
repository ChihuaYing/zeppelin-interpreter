package org.apache.zeppelin.iginx;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import java.io.*;
import java.net.*;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Enumeration;
import org.apache.zeppelin.iginx.util.HttpUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SimpleFileServer {
  private static final Logger LOGGER = LoggerFactory.getLogger(SimpleFileServer.class);

  public static String PREFIX = "/files";
  public static String PREFIX_UPLOAD = "/files/upload";
  private int port;
  private String fileDir;
  private String uploadFileDir;
  private Long uploadDirMaxSize;
  protected static final boolean isOnWin =
      System.getProperty("os.name").toLowerCase().contains("win");

  private HttpServer httpServer = null;

  public SimpleFileServer(int port, String fileDir, String uploadFileDir, long uploadDirMaxSize) {
    this.port = port;
    this.fileDir = fileDir;
    this.uploadFileDir = uploadFileDir;
    this.uploadDirMaxSize = uploadDirMaxSize;
  }

  public void start() throws IOException {
    // 检测端口是否被占用，如果占用则kill掉
    try {
      new Socket("localhost", port).close();
      LOGGER.info("SimpleFileServer started on port {}", port);
      if (isOnWin) {
        Runtime.getRuntime()
            .exec(
                "for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :"
                    + port
                    + "') do taskkill /F /PID %a");
      } else {
        Runtime.getRuntime().exec("kill -9 $(lsof -t -i:" + port + ")");
      }
    } catch (IOException e) {
      // do nothing
      LOGGER.error("restart server error.", e);
    }

    try {
      LOGGER.info("Starting SimpleFileServer on port " + port);
      httpServer = HttpServer.create(new InetSocketAddress(port), 0);
      httpServer.createContext(PREFIX, new FileHandler(fileDir));
      httpServer.createContext(PREFIX_UPLOAD, new UploadHandler(uploadFileDir));
      httpServer.start();
    } catch (IOException e) {
      LOGGER.error("Error starting SimpleFileServer", e);
      throw new RuntimeException(e);
    }
  }

  public void stop() {
    if (httpServer != null) {
      httpServer.stop(0);
    }
  }

  static class FileHandler implements HttpHandler {
    private String basePath;

    public FileHandler(String basePath) {
      this.basePath = basePath;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
      try {
        // 添加 CORS 响应头
        exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");

        // 获取请求的文件名，并构建文件路径
        String requestPath = exchange.getRequestURI().getPath();
        String fileName = requestPath.substring(PREFIX.length());
        File file = new File(basePath + fileName);

        // 检查文件是否存在且不是目录
        if (file.exists() && !file.isDirectory()) {
          // 设置响应头为文件下载
          exchange.getResponseHeaders().set("Content-Type", "application/octet-stream");
          exchange
              .getResponseHeaders()
              .set("Content-Disposition", "attachment; filename=\"" + file.getName() + "\"");
          exchange.sendResponseHeaders(200, file.length());

          // 读取文件并写入响应体
          OutputStream os = exchange.getResponseBody();
          FileInputStream fs = new FileInputStream(file);
          final byte[] buffer = new byte[0x10000];
          int count = 0;
          while ((count = fs.read(buffer)) >= 0) {
            os.write(buffer, 0, count);
          }
          fs.close();
          os.close();
        } else {
          // 如果文件不存在，返回404错误，响应体为"404 (Not Found)，可能文件已被删除，请重新执行查询“
          String response = "404 (Not Found)，可能文件已被删除，请重新执行查询";
          exchange.sendResponseHeaders(404, response.length());
          OutputStream os = exchange.getResponseBody();
          os.write(response.getBytes());
          os.close();
        }
      } catch (IOException e) {
        e.printStackTrace();
        exchange.sendResponseHeaders(500, 0); // 发送500错误
        exchange.getResponseBody().close();
      }
    }
  }

  /** upload csv file handler */
  class UploadHandler implements HttpHandler {
    private final String basePath;

    public UploadHandler(String basePath) {
      this.basePath = basePath;
    }

    @Override
    public void handle(HttpExchange exchange) {

      String zeppelinUrl = "", noteBookId = "", paragraphId = "", fileName = "";
      BufferedWriter bw = null;
      BufferedReader br = null;
      String line;
      boolean isContent = false;
      try {
        if (!"POST".equals(exchange.getRequestMethod())) {
          exchange.sendResponseHeaders(405, -1);
          return;
        }
        exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        File uploadDir = new File(HttpUtil.getCurrentPath(basePath));
        if (!uploadDir.exists()) {
          uploadDir.mkdirs();
        }
        br =
            new BufferedReader(
                new InputStreamReader(new BufferedInputStream(exchange.getRequestBody())));
        /* parse form-data */
        while ((line = br.readLine()) != null) {
          LOGGER.debug(line);
          if (isContent) {
            if (line.trim().isEmpty()) {
              break;
            }
            bw.write(line);
            bw.newLine();
          }
          if (line.startsWith("------")) {
            line = br.readLine();
            if (line.contains("filename=")) {
              fileName = line.substring(line.indexOf("filename=") + 10, line.length() - 1);
              br.readLine();
              br.readLine();
              isContent = true;
              bw =
                  new BufferedWriter(
                      new OutputStreamWriter(
                          Files.newOutputStream(new File(uploadDir, fileName).toPath())));
            } else {
              String paramName = line.substring(line.indexOf("name=") + 6, line.length() - 1);
              br.readLine();
              switch (paramName) {
                case "zeppelinUrl":
                  zeppelinUrl = br.readLine().trim();
                  break;
                case "noteBookId":
                  noteBookId = br.readLine().trim();
                  break;
                case "paragraphId":
                  paragraphId = br.readLine().trim();
                  break;
                default:
                  LOGGER.warn("unexpected params received {}", br.readLine().trim());
                  break;
              }
            }
          }
        }
        if (bw != null) {
          bw.close();
        }
        LOGGER.info(
            "received parameters:{},{},{},{}", zeppelinUrl, noteBookId, paragraphId, fileName);
        String result =
            HttpUtil.sendPost(
                String.format("%s/api/notebook/run/%s/%s", zeppelinUrl, noteBookId, paragraphId),
                null);
        LOGGER.info("result of rerun paragraph command: {}", result);
        exchange.sendResponseHeaders(200, 0);
      } catch (IOException e) {
        LOGGER.error("Error uploading file", e);
        throw new RuntimeException(e);
      } finally {
        exchange.close();
        cleanUpLoadDir();
      }
    }
  }

  /** clean earliest files when upload file director exceeds 100GB */
  public void cleanUpLoadDir() {
    new Thread(
            () -> {
              File directory = new File(uploadFileDir);
              File[] files = directory.listFiles();
              if (files != null) {
                // order by lastModified -  asc
                Arrays.sort(files, Comparator.comparingLong(File::lastModified));
                long totalSize = Arrays.stream(files).mapToLong(File::length).sum();
                for (File file : files) {
                  if (totalSize <= uploadDirMaxSize) {
                    break;
                  }
                  LOGGER.info("Deleted file {},{}", file.getAbsolutePath(), file.length());
                  totalSize = totalSize - file.length();
                  file.delete();
                }
              }
            })
        .start();
  }

  /**
   * 获取本地主机地址，普通方法会获取到回环地址or错误网卡地址，因此需要使用更复杂的方法获取
   *
   * @return InetAddress 本机地址
   */
  public static String getLocalHostExactAddress() {
    try {
      Enumeration<NetworkInterface> networkInterfaces = NetworkInterface.getNetworkInterfaces();
      while (networkInterfaces.hasMoreElements()) {
        NetworkInterface networkInterface = networkInterfaces.nextElement();
        if (networkInterface.isLoopback()
            || networkInterface.isVirtual()
            || !networkInterface.isUp()) {
          continue;
        }

        Enumeration<InetAddress> inetAddresses = networkInterface.getInetAddresses();
        while (inetAddresses.hasMoreElements()) {
          InetAddress inetAddress = inetAddresses.nextElement();
          if (!inetAddress.isLoopbackAddress()
              && !isPrivateIPAddress(inetAddress.getHostAddress())
              && inetAddress instanceof Inet4Address) {
            // 这里得到了非回环地址的IPv4地址
            return inetAddress.getHostAddress();
          }
        }
      }
    } catch (SocketException e) {
      e.printStackTrace();
    }
    return null;
  }

  // 判断是否为私有IP地址
  private static boolean isPrivateIPAddress(String ipAddress) {
    return ipAddress.startsWith("10.")
        || ipAddress.startsWith("192.168.")
        || (ipAddress.startsWith("172.")
            && (Integer.parseInt(ipAddress.split("\\.")[1]) >= 16
                && Integer.parseInt(ipAddress.split("\\.")[1]) <= 31));
  }
}
