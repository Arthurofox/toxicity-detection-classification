# Notebook for our work

# first part : data analysis and pre-processing

(venv) arthursavary@Arthurs-MacBook-Pro toxicity-detection-classification % cd src 
(venv) arthursavary@Arthurs-MacBook-Pro src % python3 pre_analysis.py 
Error: File data/train.csv not found. Please update the path.
(venv) arthursavary@Arthurs-MacBook-Pro src % python3 pre_analysis.py
Loading datasets...
Combining datasets for analysis...

=== COMBINED DATASET ANALYSIS ===

=== Dataset Structure ===
Columns: ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

=== Data Types ===
id               object
comment_text     object
toxic             int64
severe_toxic      int64
obscene           int64
threat            int64
insult            int64
identity_hate     int64
dtype: object

=== Missing Values ===
Series([], dtype: int64)

=== Sample Data ===
                 id                                       comment_text  toxic  severe_toxic  obscene  threat  insult  identity_hate
0  0000997932d777bf  Explanation\nWhy the edits made under my usern...      0             0        0       0       0              0
1  000103f0d9cfb60f  D'aww! He matches this background colour I'm s...      0             0        0       0       0              0
2  000113f07ec002fd  Hey man, I'm really not trying to edit war. It...      0             0        0       0       0              0

=== Label Distribution ===
Label columns found: ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

TOXIC
toxic
 0    202165
-1     89186
 1     21384
Name: count, dtype: int64
Percentage positive: 6.84%

SEVERE_TOXIC
severe_toxic
 0    221587
-1     89186
 1      1962
Name: count, dtype: int64
Percentage positive: 0.63%

OBSCENE
obscene
 0    211409
-1     89186
 1     12140
Name: count, dtype: int64
Percentage positive: 3.88%

THREAT
threat
 0    222860
-1     89186
 1       689
Name: count, dtype: int64
Percentage positive: 0.22%

INSULT
insult
 0    212245
-1     89186
 1     11304
Name: count, dtype: int64
Percentage positive: 3.61%

IDENTITY_HATE
identity_hate
 0    221432
-1     89186
 1      2117
Name: count, dtype: int64
Percentage positive: 0.68%

=== Label Co-occurrence ===

=== Text Length Analysis (comment_text) ===

Character count statistics:
count    312735.000000
mean        379.773262
std         591.767791
min           1.000000
25%          87.000000
50%         193.000000
75%         414.000000
max        5000.000000
Name: char_count, dtype: float64

Word count statistics:
count    312735.000000
mean         64.500145
std          99.138334
min           0.000000
25%          15.000000
50%          34.000000
75%          71.000000
max        2321.000000
Name: word_count, dtype: float64

Average text length by toxicity:
toxic
-1    351.601047
 0    402.691178
 1    280.604097
Name: char_count, dtype: float64
toxic
-1    59.444386
 0    68.415161
 1    48.573466
Name: word_count, dtype: float64

=== Vocabulary Analysis ===

Top 20 words overall:
article: 105723
wikipedia: 83699
page: 79164
talk: 53805
like: 52864
just: 52227
don: 44040
think: 38522
fuck: 37443
people: 34685
know: 34490
edit: 31591
use: 29904
articles: 29336
time: 28600
did: 26709
thanks: 24625
make: 23765
good: 23538
ve: 23350

No demographic columns found in the dataset.

=== Sample Examples ===

TOXIC EXAMPLES:
1. I am going to kill you. I am going to murder you.
2. And the pay sucks o)
3. I think you faggots should stop trying to be the first one to put bullshit information on here, and instead wait for the news organizations to get their facts straight.
4. Yeah your brain, you pathetic moron. Go get an education first. Don't come back here until then. See you in about five years. If you have the mediocre IQ you seem to have. It's got to do with EBCDIC. ...
5. :You might like to consider that I don't give a shit what you do or think.

NON-TOXIC EXAMPLES:
1. "
The only reason I would oppose that is because it would not be consistent with all other album articles. And also, making it smaller would also increase the 'emptyness' feeling as there is already t...
2. " 

 == ""Mentos and coke"" experiment explanation == 

 I very much doubt that the surface activity of gum arabic has anything to do with its ability to nucleate carbon dioxide release from carbonate...
3. ::: Nevertheless, 48 hours is too much. And again, it's a talk page. The best solution would be to just ignore this above paragraph about Stalin, I would think. Other comments above seem more construc...
4. OK please talk to me 

please talk to me i want to show you shane duffy is a real player
5. ", you're infringing owner B. However if picture A is out of copyright you can take your picture C of it, and picture C is your copyright. Which I think is why art galleries ban cameras - they want to...


=== TRAINING DATASET ANALYSIS ===

=== Dataset Structure ===
Columns: ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

=== Data Types ===
id               object
comment_text     object
toxic             int64
severe_toxic      int64
obscene           int64
threat            int64
insult            int64
identity_hate     int64
dtype: object

=== Missing Values ===
Series([], dtype: int64)

=== Sample Data ===
                 id                                       comment_text  toxic  severe_toxic  obscene  threat  insult  identity_hate
0  0000997932d777bf  Explanation\nWhy the edits made under my usern...      0             0        0       0       0              0
1  000103f0d9cfb60f  D'aww! He matches this background colour I'm s...      0             0        0       0       0              0
2  000113f07ec002fd  Hey man, I'm really not trying to edit war. It...      0             0        0       0       0              0

=== Label Distribution ===
Label columns found: ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

TOXIC
toxic
0    144277
1     15294
Name: count, dtype: int64
Percentage positive: 9.58%

SEVERE_TOXIC
severe_toxic
0    157976
1      1595
Name: count, dtype: int64
Percentage positive: 1.00%

OBSCENE
obscene
0    151122
1      8449
Name: count, dtype: int64
Percentage positive: 5.29%

THREAT
threat
0    159093
1       478
Name: count, dtype: int64
Percentage positive: 0.30%

INSULT
insult
0    151694
1      7877
Name: count, dtype: int64
Percentage positive: 4.94%

IDENTITY_HATE
identity_hate
0    158166
1      1405
Name: count, dtype: int64
Percentage positive: 0.88%

=== Label Co-occurrence ===

=== Text Length Analysis (comment_text) ===
^A
Character count statistics:
count    159571.000000
mean        394.073221
std         590.720282
min           6.000000
25%          96.000000
50%         205.000000
75%         435.000000
max        5000.000000
Name: char_count, dtype: float64

Word count statistics:
count    159571.000000
mean         67.273527
std          99.230702
min           1.000000
25%          17.000000
50%          36.000000
75%          75.000000
max        1411.000000
Name: word_count, dtype: float64

Average text length by toxicity:
toxic
0    404.549339
1    295.246044
Name: char_count, dtype: float64
toxic
0    68.967874
1    51.289787
Name: word_count, dtype: float64


=== TEST DATASET ANALYSIS ===

=== Dataset Structure ===
Columns: ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

=== Data Types ===
id               object
comment_text     object
toxic             int64
severe_toxic      int64
obscene           int64
threat            int64
insult            int64
identity_hate     int64
dtype: object

=== Missing Values ===
Series([], dtype: int64)

=== Sample Data ===
                 id                                       comment_text  toxic  severe_toxic  obscene  threat  insult  identity_hate
0  00001cee341fdb12  Yo bitch Ja Rule is more succesful then you'll...     -1            -1       -1      -1      -1             -1
1  0000247867823ef7  == From RfC == \n\n The title is fine as it is...     -1            -1       -1      -1      -1             -1
2  00013b17ad220c46  " \n\n == Sources == \n\n * Zawe Ashton on Lap...     -1            -1       -1      -1      -1             -1

=== Label Distribution ===
Label columns found: ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

TOXIC
toxic
-1    89186
 0    57888
 1     6090
Name: count, dtype: int64
Percentage positive: 3.98%

SEVERE_TOXIC
severe_toxic
-1    89186
 0    63611
 1      367
Name: count, dtype: int64
Percentage positive: 0.24%

OBSCENE
obscene
-1    89186
 0    60287
 1     3691
Name: count, dtype: int64
Percentage positive: 2.41%

THREAT
threat
-1    89186
 0    63767
 1      211
Name: count, dtype: int64
Percentage positive: 0.14%

INSULT
insult
-1    89186
 0    60551
 1     3427
Name: count, dtype: int64
Percentage positive: 2.24%

IDENTITY_HATE
identity_hate
-1    89186
 0    63266
 1      712
Name: count, dtype: int64
Percentage positive: 0.46%

=== Label Co-occurrence ===

=== Text Length Analysis (comment_text) ===

Character count statistics:
count    153164.000000
mean        364.875121
std         592.492099
min           1.000000
25%          79.000000
50%         180.000000
75%         392.000000
max        5000.000000
Name: char_count, dtype: float64

Word count statistics:
count    153164.000000
mean         61.610751
std          98.959698
min           0.000000
25%          14.000000
50%          31.000000
75%          67.000000
max        2321.000000
Name: word_count, dtype: float64

Average text length by toxicity:
toxic
-1    351.601047
 0    398.059995
 1    243.833333
Name: char_count, dtype: float64
toxic
-1    59.444386
 0    67.037607
 1    41.751888
Name: word_count, dtype: float64

Analysis complete! Plots saved to the 'plots' directory.
(venv) arthursavary@Arthurs-MacBook-Pro src % 