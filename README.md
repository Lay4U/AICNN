모델은 keras github에 있는 example에서 따왔습니다.

다른 팀들도 비슷한 모델을 사용할 것이니 성능 척도에 기준을 두고 업데이트 하도록 하는게 좋을것 같습니다.

2)  모델 학습 경우에는 학습 데이터 변형 등과 같이 추가적인 학습 방법은 불가능합니다. 그 외에 decay 같은 optimizer의 파라미터는 수정하셔도 됩니다.
     모델 구현 부분에서는 BatchNormalization 외에는 다 사용 가능합니다.
3) 성능 평가 기준은 최종 epoch에서의 accuracy입니다.

성능 평가 기준은 최종 epoch에서의 accuracy이고, accuracy 갱신할때마다 push 해주세요 

갱신 하지 못하더라도 새로운 아이디어 생각나실때마다 main말고 다른 이름으로 push 해주세요

큰틀에서 은닉층 설계랑 잘 설계된 모델에서 하이퍼파라미터 갱신 이런식으로 진행해야 될것 같네요

나머지 정보들은 Wiki에서 공유합니다
