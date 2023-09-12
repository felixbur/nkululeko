# process_database.py --> PORTUGUESE
# modified from https://github.com/Strong-AI-Lab/emotion/blob/master/datasets/Portuguese/process.py


import argparse
import re
from pathlib import Path

import pandas as pd

REGEX = re.compile(r"^\d+[sp][AB]_([a-z]+)\d+$")
from sklearn.model_selection import train_test_split

emotion_map = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "sad": "sadness",
    "neutral": "neutral",
    "surprise": "surprise",
}

sentence_map = {
    # Sentences
    "estaMesa": "Esta mesa é de madeira",
    "oRadio": "O rádio está ligado",
    "aqueleLivro": "Aquele livro é de história",
    "aTerra": "A Terra é um planeta",
    "oCao": "O cão trouxe a bola",
    "eleChega": "Ele chega amanhã",
    "estaRoupa": "Esta roupa é colorida",
    "osJardins": "Os jardins têm flores",
    "asPessoas": "As pessoas vão a concertos",
    "haArvores": "Há árvores na floresta",
    "osTigres": "Os tigres são selvagens",
    "oQuadro": "O quadro está na parede",
    "alguemFechou": "Alguém fechou as janelas",
    "osJovens": "Os jovens ouvem música",
    "oFutebol": "O futebol é um desporto",
    "elaViajou": "Ela viajou de comboio",
    # Pseudo-sentences
    "estaDepa": "Esta dêpa é de faneira",
    "oDarrio": "O dárrio está guilado",
    "aqueleJicro": "Aquele jicro é de hisbólia",
    "aPirra": "A Pirra é um flaneto",
    "oLao": "O lão droube a nóma",
    "eleChena": "Ele chena aguinhã",
    "estaSouda": "Esta souda é lacoripa",
    "osBartins": "Os bartins têm pléres",
    "asSemoas": "As semoas vão a cambêrtos",
    "haArjuques": "Há árjuques na plurisca",
    "osLagres": "Os lagres são siltávens",
    "oJuadre": "O juadre está na pafêne",
    "alguemBelhou": "Alguém belhou as jalétas",
    "osDofens": "Os dófens mavem tézica",
    "oDutebel": "O dutebel é um nesforpo",
    "elaJiavou": "Ela jiavou de lantóio",
}

sentence_type_map = {
    "s": "sentence",
    "p": "pseudo-sentence",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='PORTUGUESE')
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    paths = list(data_dir.glob('*.wav'))

    sent_info = pd.concat(
        [
            pd.read_csv(
                data_dir / x,
                sep="\t",
                skiprows=5,
                header=0,
                index_col=0,
            )
            for x in [
                "Castro_2010_AppxB_Sents.txt",
                "Castro_2010_AppxB_Pseudosents.txt",
            ]
        ],
        ignore_index=True,
    ).set_index("Stimulus ")

    emotions = [emotion_map[REGEX.match(p.stem).group(1)] for p in paths]
    speakers = [p.stem[p.stem.find("_") - 1] for p in paths]
    intensity = sent_info.loc[[p.stem for p in paths],
                              "Intensity (1-7)"].tolist()
    sent_type = [sentence_type_map[p.stem[p.stem.find("_") - 2]] for p in paths]
    language = ["portuguese" for p in paths]

    df = pd.DataFrame({"file": paths, "emotion": emotions, "speaker": speakers,
                       "intensity": intensity, "sentence_type": sent_type,   
                       "language": language})

    # split into train and test
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, 
                                         stratify=df['emotion'])
    
    # save to CSV
    df.to_csv(output_dir / 'portuguese.csv', index=False)
    df_train.to_csv(output_dir / 'portuguese_train.csv', index=False)
    df_test.to_csv(output_dir / 'portuguese_test.csv', index=False)

    print(f"Portuguese: {len(df)} samples, {len(df_train)} train, {len(df_test)} test")
    

if __name__ == '__main__':
    main()



