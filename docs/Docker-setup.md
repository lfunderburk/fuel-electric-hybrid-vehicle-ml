### Docker setup

```
docker build -t <application>:latest -f <Dockerfile-name> .
docker tag <application>:latest lgfunderburk/<application>:latest
docker push lgfunderburk/<application>:latest
```

Where

`<application>` is one of `myapi`, `mydashapp`, `mypipeline` and
`<Dockerfile-name>` is one of `Dockerfile.api`, `Dockerfile.dash` and `Dockerfile.pipe`