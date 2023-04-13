# 日本人顔画像を外国人風に変換するアプリケーション
- ８種類のスタイルから任意のものを選択できます
- 初回実行時のみ入力画像の潜在変数探索を行うため、実行に時間を要します
- [StyleCariGAN](https://github.com/wonjongg/StyleCariGAN)の実装をベースにしています。

## 環境構築
Docker, docker-composeが使用可能であることを前提としています。

- コンテナ作成
```bash
make build
```
- コンテナ起動
```bash
make up
```

## 実行方法
コンテナが実行されると同時にGradioアプリケーションが立ち上がります。
```bash
make exec
```
