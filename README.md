## nanoGPT-rs

A simple rust implementation of [nanoGPT](https://github.com/karpathy/nanoGPT) with the formidable crate [dfdx](https://github.com/coreylowman/dfdx).

I have adjusted the model parameters to fit them within the memory constraints of my older 2GB GPU.


Train on Shakespeare dataset(1.1M)
```bash
#cargo run --release --features cuda

God#ò]##¹^ª½ûë©####ÃÊ½sSml?##^ª'##½o#ôrG####'á¶o#¹$É¶#Êôâ###?l#?#$»#ÄÈë]###nÊ#Ñ{Ñúë#¹#¹Ê#ãâ'ü?'K#SK|]ã##ý]'¶##n##½###|#â#Ñ?#?ã½Ê$¹'S#¶ÈÈ]áÀ?Ño#á¹#á{?'#;ú#¶#]á¹##Sº.ü#ÄrÑ
Epoch 0, loss 1453.52856, gen: God Irerse reancol ave#Co p f wico s nd tupicer mes ho t whand te upthind stie ors be#Thin tourer: be de,#An me g ththe be thenthearistont, t the cupre t, mothe bos thin tend be tan ick t ththithone beshaicongh I by t athe the tho cesu he toreit.#####SPORY
Epoch 1, loss 1182.35461, gen: God poth pold poppelt of eem.###SELUCENCE:#O gooooo yourt.##KING LEG VI:#To not as botininy for ass we you,#As by hold hear his shal.##CATIO:#Entange speel of my liked,#A and the reater it an strents#Mand your and hinght stake and wind.##CAMINE:#You I dor
Epoch 2, loss 982.88208, gen: God,#That you dike and so not the a the give#To to stard hatth it briace of that#Of remars to better.##First Clowardin:#Our again with the rong art the perfortience#To the dreange and that no the is a thought,#Firenciment which die, though mainiot is sende
```

Train on [Tinystories dataset(valid)(19M)](https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt)
```bash
Once upon a time, there was a horse with many fish and he had felt happy. Suddenly, the ant admired his mommy, so he decided to help his man said yes. The car was very angry and made the veterinarian decistened to play for the animals.#All the started to r
```
Only 1 epoch!



