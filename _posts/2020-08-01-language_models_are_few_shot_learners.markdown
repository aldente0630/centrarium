---
layout: post
title: OpenAI GPT-3 언어 모델은 퓨-샷(Few-Shot) 학습자이다
date: 2020-08-01 00:00:00
author: Soheil Tehranipour
categories: Data-Science
---  
  
  
**Soheil Tehranipour의 [*OpenAI GPT-3: Language Models are Few-Shot Learners*](https://medium.com/analytics-vidhya/openai-gpt-3-language-models-are-few-shot-learners-82531b3d3122)을 번역했습니다.**
  
  
- - -
  
![GPT-3는 처음에 많은 걸 이야기한다.](https://aldente0630.github.io/assets/language_models_are_few_shot_learners1.jpeg)
   
[OpenAI](https://openai.com)는 최근 자연어 처리를 위한 심층 학습 모델 [GPT-3](https://github.com/openai/gpt-3) 설명을 담은 논문을 발표했다. GPT-3는 이전 버전인 GPT-2보다 파라미터 개수가 무려 **100배** 많은 **175억 개**(!!!)이다. 해당 모델의 경우 거의 5조 개에 가까운 단어들로 사전 훈련을 수행했고 미세 조정 없이도 여러 NLP 벤치 마크에서 SOTA 성능을 달성했다.

## 다시 한번 말하지만 “미세 조정 없이”

[논문](https://github.com/openai/gpt-3)을 보면 **GPT-3**은 **BERT** 같은 디노이징 오토인코더(denoising autoencoder)가 아닌, 자기 회귀 언어 모델이다. 두 언어 모델 아키텍처 간의 차이점에 대해 글을 써보기로 마음먹었다. 해당 논문은 거대한 언어 모델로 무엇을 할 수 있는지에 관한 조사 자료이다. 이 언어 모델은 다른 사람이 만든 언어 모델보다 규모가 훨씬 더 크다.
  
*OpenAI 연구원: “우리는 주어진 작업에 의존성을 갖지 않는, 보편적 성능에 중점을 두었기 때문에 GPT-3을 미세 조정하지 않았습니다.”*

![비교: 모델 간 파리미터 크기.](https://aldente0630.github.io/assets/language_models_are_few_shot_learners2.png)
## GPT-3은 파라미터 개수 1억 2,500만 개에서 175억 개까지 8개의 서로 다른 크기로 제공된다.
* 가장 큰 GPT-3 모델을 보면 이전 기록 보유자인 [T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)(11억 개)와 [튜링-NLG](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)(17억 개)보다 규모가 훨씬 더 크다.
* 가장 작은 모델은 [ALBERT-기본](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html)인데 위 차트에 그 크기가 나와있다.
  
# 당신에게 필요한 건 언어 모델 뿐.
따라서 자연어 생성에 적합해 보인다. 내가 여기서 주로 살펴보고 싶은 건 학문적인 성취이다. Microsoft, Google 또는 OpenAI 같은 최첨단 기술 회사만 접근 가능한 GPT-3를 우리에게 유용한 상용 유틸리티에 적용해보는 일은 불가능(적어도 현재는 불가능)하다.

# 더 파고 들기 전에 지금까지 있었던 일들을 살펴보자…

* 신경망을 적용하면서 NLP에 엄청나게 많은 발전이 있어왔으며 이는 2013년 [Tomas Mikolov](https://scholar.google.com/citations?user=oBu8kMMAAAAJ&hl=en)가 [Word2vec](https://arxiv.org/abs/1301.3781)를 도입하면서 시작됐다. Word2Vec은 단어에 대해 문맥에 따라 자기 지도 방식으로 학습한 분산 표현이다.

(번역 중) 
