# イベント表本時刻の定義訂正

最初の計画と実行はon-disk first=1を時刻にも適用しarray258を候補eventとしました。
CSV ITEM259がarray258に一致することは空間値と配列番号の対応を示すだけで、
event1.295秒がITEM259という時間対応は証明しませんでした。この推定は撤回します。

[C3D公式 EVENT:TIMES](https://www.c3d.org/HTML/Documents/eventtimes1.htm)は
first3Dsample(frame1)=0.0秒と定義します。この規約ならarray259/sourceframe260です。
元のproducerがlegacyなframe/rate時刻を使った可能性はこの資料だけでは排除できません。
実際の接触解放を画面から独立確認していないためsource_event_mapping=UNKNOWNを維持します。

この訂正を計算前に別途commitし、標準規約のarray259をspec nominal、array258を
legacy alternativeとして両方の結果を残します。元の計画/実行summaryとcodehashは
initial_mapping/に保持します。規約は予測誤差で選びません。元のshift0/1と新しい
shift-1/0は同じ二つの物理標本窓です。数値のよいほうを採用しません。

これは時刻定義の訂正で、真の離脱時刻を検証したことではありません。既存OBPや
Primeの時刻対応に自動適用せず、当該資料のsourceと規約だけを扱います。
