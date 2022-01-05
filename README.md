# 도리도리 카운터

## Overview
도리도리 카운터는 사람의 인중 움직임을 감지하여 얼마나 고개를 흔드는지 알 수 있는 프로그램입니다. 

## Sample Video
[유튜브 링크](https://youtu.be/8zCk4WtGvVY)

## QuickStart

### Install

해당 repo를 clone하시고 tutorial.ipynb 파일에 적힌 대로 이용하시면 됩니다. 

### Usage Example

```
    from doridori import Doridori
    dori = Doridori(filepath)
    dori.detect_face()
    dori.fit()
``` 

위의 코드로 영상의 도리도리 횟수를 구할 수 있습니다. 

```
    dori.save_video(filepath)
```

위의 코드로 실시간으로 도리도리 그래프를 보여주는 영상을 filepath 경로에 저장할 수 있습니다. 

## Contacts
코드 및 기타 문의는 github issue나 dong@whew.ai로 부탁드립니다. 감사합니다.
