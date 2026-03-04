# Chess-Competition

This is a C++ chess competition base repository. Here we are going to host only the chess engine part of the competition. The competition will be held on a web platform, and the source code will be available on a different repository.

> **Note:** the web UI supports a "recent bots" filter. Bots are tagged with the
> timestamp of their last GitHub activity and students can choose to run
> tournaments using either **all bots** or only those updated within the past
> six months. The build scripts capture this `updated_at` field and even allow
> optional filtering via `RECENT_ONLY=1`.


## Main activity

You have to code a chess engine from the ground up. You will receive a string as input following [FEN](https://www.chess.com/terms/fen-chess) notation and the output should be the move as [UCI](https://en.wikipedia.org/wiki/Universal_Chess_Interface). 

## Rules

- Implement your chess engine only inside the chess-bot folder in C++;
- Create .h and .cpp files inside chess-bot;
- Obey the interface specified on chess-bot;
- You might want to test your code via terminal via chess-cli, or chess-gui;
- Merge requests are welcome;
- When you submit your code, you should zip only the contents of the chess-bot folder and send it to the system;
- Do not use sub-folders inside the chess-bot folder, it will break my automation;

## Folder structure

- chess-bot: Here you will implement your chess engine;
- chess-validator: Here you will find the chess-validator code;
- chess-gui: Here you will find the chess-gui code;

## How the competition will work

- A tournament will run every day;
- The tournament consists of making all players playing against each other on both sides of the board;
- You may use up to 16GB ram and 12 cores of CPU;
- Each turn should take less than 10 seconds;
- You are not allowed to use any external library besides the ones already provided here;
- You can use AI assisted tools, but you have to state it on your code, and you will receive a penalty on your final score of 20%;
- You may work in teams of up to 2 people;
- Your username should not match your real name by any means in order do follow FERPA compliance;

## Web and API

The online version will be hosted on https://web.gameguild.gg

The source code will be coded here https://github.com/InfiniBrains/website

## Third Parties

- [Disservin/chess-library](https://github.com/Disservin/chess-library)
- [libsdl-org](https://github.com/libsdl-org)
- [ocornut/ImGUI](https://github.com/ocornut/imgui)
- [cpm-cmake/cpm.cmake](https://github.com/cpm-cmake/CPM.cmake)
- [Neargye/magic_enum](https://github.com/Neargye/magic_enum)

## Licences

[Check it out](third_party.txt)