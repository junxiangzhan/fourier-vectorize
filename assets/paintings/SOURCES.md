# Painting sources

Only public-domain images are kept in this repository.

- `adam.jpg`: Michelangelo, *The Creation of Adam* (c. 1512), Sistine Chapel.
  Public domain worldwide.
- `gleizes.jpg`: Albert Gleizes, *Femme et enfant* (Woman and child), 1920.
  The artist died in 1953: public domain where copyright lasts 70 years
  after the author's death (since 2024), and in the United States as a work
  from before 1929. Original file name
  `Albert_Gleizes,_1920,_Femme_et_enfant_(Woman_and_child).jpg`.

Put any other picture as a `.jpg` in this folder and decode it with

    python scripts/decode_paintings.py assets/paintings

then run the showcase on every BMP here, or on one picture:

    cargo run --release -- all --paintings assets/paintings
    cargo run --release -- paint assets/paintings/adam.bmp --factor 4
