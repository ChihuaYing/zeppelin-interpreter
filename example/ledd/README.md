# SIGMOD Demo 使用说明

请确保网络畅通，必要时使用代理。

## 构建

构建 zeppelin-interpreter 和 schemapile-loader

```shell
mvn -f ../../pom.xml clean package
mvn -f schemapile-loader/pom.xml clean package
```

构建并运行容器

```shell
docker compose up --build
```

## 使用示例

在 IGinX 容器启动后，查看日志确认一下步骤完成：
- 等待 SchemaPile 数据集导入到 IGinX 中，可以通过 `Inserting Data` 进度条确认进度。
- 等待 执行 `store_embedding` 函数的 transform job 第一次执行完成，可以通过 `Inserting embeddings` 进度条确认进度。

为了避免 Milvus 出现 Bug，可以在上述步骤完成后单独重启一次 Milvus 容器。

打开 http://localhost:8080 进入 zeppelin 页面，打开示例笔记本执行示例语句。部分语句执行时间较长，可通过 IGinX 日志确认进度。




