# Qualitative caption examples (GPT-2, T5 -- baseline greedy)

_Captions shown trimmed at the last full sentence for readability; all metrics in the tables are computed on the raw untrimmed output._


## Successes (semantic hits)

### Clip 294

**Reference:** instrumental rock music with a drum kit, lots of cymbal crashes, acoustic guitar strumming, harmonized lead guitars and an active bass guitar.

**GPT-2:** This is a rock music piece. There is a male vocalist singing melodically. The electric guitar is playing the main tune while the bass guitar is playing in the background. The rhythmic background consists of a simple acoustic drum beat. The atmosphere is groovy.

**T5:** This is a rock music piece. There is a male vocalist singing melodically. The electric guitar is playing a simple tune with a bass guitar playing in the background. The rhythmic background consists of a fast-paced acoustic drum beat.

_Why it works:_ Instrumental rock: both models nail genre (rock) + instruments (electric guitar, bass, acoustic drums) + groovy/energetic mood.

### Clip 219

**Reference:** This is a drum & bass electronic music piece. There is no singer. There is a strong and fuzzy feeling bass sound. The rhythmic background consists of a loud and high tempo electronic drum beat. The atmosphere is modern and energetic. This piece could be playing at a nightclub or a dance club.

**GPT-2:** This is a techno music piece. There is a male vocalist singing melodically. The melody is being played by the synth bass. The rhythmic background consists of a fast-paced electronic drum beat. The atmosphere is groovy. This piece could be used in the background of nightclubs.

**T5:** This is a techno house music piece. There is a male vocalist singing melodically. The melody is being played by the synth bass while the rhythmic background consists of a fast-paced electronic drum beat. The atmosphere is energetic.

_Why it works:_ Drum & bass / electronic: both land the electronic family (techno/house), synth bass, fast electronic beat, and the energetic nightclub setting.

### Clip 38

**Reference:** The low quality recording features a rock song that consists of a flat male vocal, alongside background male vocals, singing over shimmering hi hats, groovy bass, punchy kick and snare hits and electric guitar melodies. It sounds energetic and emotional.

**GPT-2:** The low quality recording features a rock song that consists of a flat male vocal, alongside harmonizing male vocals, singing over punchy kick and snare hits, shimmering hi hats, groovy bass guitar, punchy snare and shimmering cymbals. It sounds energetic and exciting.

**T5:** This is a rock music piece. There is a male vocalist singing melodically in the lead. The electric guitar is playing a simple tune with a bass guitar playing in the background. The rhythmic background consists of a fast-paced acoustic drum beat.

_Why it works:_ Rock with flat male vocal: GPT-2 closely mirrors the reference's texture -- flat + harmonizing male vocals, hi-hats, kick/snare, groovy bass, energetic.

### Clip 273

**Reference:** A female singer sings this emotional melody. The song is medium tempo with guitar strumming rhythm, strong bass line, steady drumming rhythm, keyboard accompaniment and sound effects. The song is emotional and romantic. The song is a euro pop love song and has poor audio quality.

**GPT-2:** A female vocalist sings this soft melody. The tempo is medium with a keyboard accompaniment, steady drumming, steady bass line and keyboard accompaniment. The song is mellow, soft, mellifluous, emotional, sentimental, melancholic and romantic.

**T5:** This is a mellow, soft rock piece. There is a male vocalist singing melodically. The melody is being played by the electric guitar while the bass guitar is playing in the background. The rhythm is provided by a simple acoustic drum beat.

_Why it works:_ Emotional euro-pop: GPT-2 correctly identifies a FEMALE vocalist, medium tempo and emotional/romantic mood (T5 mis-genders as male).

### Clip 86

**Reference:** The low quality recording features a live performance of a jazz song and it consists of a groovy bass guitar, groovy rhodes keys, electric guitar melody, shimmering hi hats, punchy kick and punchy snare hits. It sounds groovy, funky and passionate. The recording is noisy and in mono.

**GPT-2:** This is a jazz music piece. There is a saxophone solo being played. The melody is being played by the piano and the electric guitar. The rhythmic background consists of a simple acoustic drum beat. The atmosphere is groovy. This piece could be used in the background of a jazz bar.

**T5:** This is a live recording of a jazz music piece. There is a saxophone playing the lead melody while a bass guitar is playing in the background. The rhythm is being played by the acoustic drums. The atmosphere is jazzy.

_Why it works:_ Jazz: both recover the jazz genre, groovy mood, and guitar/bass/drums; GPT-2 adds a plausible saxophone lead.

### Clip 313

**Reference:** A male vocalist sings this spirited song. The tempo is fast with intense drumming, loud harmonic electric guitar feedback, bass guitar , emphatic vocals and keyboard accompaniment. The song is a loud, progressive, powerful youthful and emphatic Hard Rock song.

**GPT-2:** A male vocalist sings this spirited Rock song. The tempo is medium fast with keyboard harmony, steady drumming, groovy bass lines, keyboard accompaniment and vocal backup. The song is youthful, energetic, enthusiastic, enthusiastic, enthusiastic and electrifying. This song is a Hard Rock/Heavy Metal/Rock.

**T5:** This is a rock music piece. There is a male vocalist singing melodically in the lead. The electric guitar is playing a simple tune with a bass guitar playing in the background. The rhythmic background consists of a fast-paced acoustic drum beat.

_Why it works:_ Hard rock: GPT-2 paraphrases the reference well -- spirited rock, fast tempo, keyboard + bass, youthful/energetic.


## Failure cases

### Clip 3

**Reference:** The low quality recording features a printer that produces printing sounds and some muffled music playing in the background, probably on a radio. It is noisy and in mono.

**GPT-2:** This is a recording of a live performance of a pop music piece. There is a male vocalist singing melodically. The melody is being played by a keyboard and a bass guitar while the rhythmic background is provided by a fast tempo electronic drum beat.

**T5:** This is a live recording of a DJ performance. The DJ is scratching the turntable repeatedly. The turntable is a low quality recording. The sound of the turntable is distorted and the recording is of very poor quality.

_Failure mode:_ Non-music audio (a printer + muffled background radio). Both models confidently hallucinate a full music piece -- the encoder/pipeline assumes music is present.

### Clip 47

**Reference:** The low quality recording features a metallic hit, followed by resonating bowl tone. The recording is mono and noisy.

**GPT-2:** This is a meditation music piece. There is a bowl-like sound being played by the bowl. The bowl is being rubbed repeatedly. There is a calming and relaxing atmosphere to this piece. This piece can be used in the background of meditation videos. It can also be used in the background of meditation videos.

**T5:** This is a Tibetan singing bowl playing in the background of a meditation video. There is a chanting bowl being played. There is a chanting bowl being played. There is a chanting bowl being played. There is a chanting bowl being played.

_Failure mode:_ Metallic hit + singing-bowl tone: greedy decoding degenerates into verbatim repetition ('a chanting bowl being played' x4). Motivates the decoding ablations.

### Clip 25

**Reference:** A solo trumpet playing an energetic staccato melody with dramatic glissandi. The trumpet has been processed to add a stereo delay effect and reverb. The tone is very bright.

**GPT-2:** This is a live performance of a classical music piece. There is a brass section playing a simple tune while the wind section is playing a simple tune. The atmosphere is lively. This piece could be used in the background of a movie scene in the desert.

**T5:** This is a classical music piece. There is a string section playing a groovy tune. The brass section is playing a groovy bass line. The atmosphere is lively.

_Failure mode:_ Solo trumpet: both over-elaborate into full classical brass/string sections and lose the key fact that it is a single solo instrument.
