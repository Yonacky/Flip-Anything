# 多项目工作环境

用 `context.ps1 list` 查看项目；每次显式 `context.ps1 context --project <id>` 选项目，没有隐式全局活动项目。当前项目为 flip_anything。

projects/<id>/project.json 只绑定目标、一个明确环境版本、项目经验卡片和输出位置；BRIEF.md 给 DS 当前任务指导。共享设备的事实放 environments/<environment>/<version>/；项目经验放 knowledge/<project>/。多个项目可以显式引用同一个环境版本，不自动共享经验，不同时控制同一设备。

新增项目由维护者复制现有 project.json/BRIEF.md 的结构并填写独立 id、目标、输出位置和经验列表（可为空）。新环境版本先完成测量资料与组件状态，再更新具体项目绑定，保留旧版本。更新后重建 context-release.json 并离线运行 check；通过后只同步清单文件。不要按文件日期自动挑选“最新”标定。

原运行代码、设备门槛和证据路径保持兼容。projects/environments/knowledge 是新资料入口；flip_pipeline/ds_control 是运行实现；logs 是原始证据；workspace/projects 是一次任务的分析材料。研究文档和旧实现按需查看，DS 不必遍历整个仓库。

模板位于 _template/project.json.example 与 _template/BRIEF.md。维护者完成资料审核后运行 `.venv-pipeline/Scripts/python.exe scripts/publish_context.py --release <版本ID>` 重建资料清单，再用 context.ps1 check 验证每个项目。该脚本只生成资料哈希，不发布设备标定或执行代码；DS 不运行维护发布脚本。更改受 source-release.json 管理的文件还需要维护者完成独立代码发布，否则执行器会拒绝启动。
