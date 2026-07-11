Notes written by CO to be addressed later 

- go through all TODO notes in the repo
- should EmbedDP really be in ET? Or renamed? how much of it's functionality actually relies on DP? Same questions for dp_transform I think? 
- should EdgeEmbed have a simpler version that acts on individual edges and then broadcasts automatically? And maybe this should be part of the `graph` folder rather than `embedding` folder? 
- pooling could logically be moved into embedding if the "embedding" folder survives, it can be interpreted in two ways: (i) embedding of a particle system; (ii) message operation
- how many of the utils should be moved out of ET into a lib/ETUtils
- where is set_product used? 
