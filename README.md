# Tetris Sovler
Training an AI to play Tetris using genetic algorithm

# show video and center it

<video src='demo_vid.mp4' text-align='center' width='25%' height='25%' controls preload></video>

## Introduction

The main goals of this project is for me to build a functional game of Tetris using matplotlib's animation capabilities as well as training a simple AI to survive as long as it can in the game using genetic algorithm. Due to the extensive amount of time it takes for the bot to play each game, the training will be done through multiprocessing using my local machine with a 11th gen Intel Core i9-11900k @ 3.5ghz, with 8 cores and 16 logical processes. 

## Motivation
As a fan of old classic video games, Tetris has played a key role in my childhood. I thought it would be quite a challenging coding practice for myself to build it from scratch. With that thought, the data scientist part of me began to wonder about the possibility to create an AI to dominate this game in ways I never could as a child. Tetris has long been proven to be an NP-Hard problem and has been a very popular challenge for AI practitioners to conquer. Given that each move in Tetris creates so many different outcomes,  the problem spans across a very large set of discrete state space, which makes Genetic Algorithm (GA) an execellent optimization method over most ML algorithms. Furthermore, GA has great parallelizability which will be proven to be tremendously helpful in speeding up the training process. 





