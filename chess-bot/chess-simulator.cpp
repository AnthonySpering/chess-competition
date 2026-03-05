#include "chess-simulator.h"
#include "chess.hpp"
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <random>
#include <array>

using namespace ChessSimulator;

static constexpr int INF = 1000000;
static constexpr int MATE_SCORE = 100000;
static constexpr int MAX_DEPTH = 64;
static uint8_t currentAge = 0;
static uint64_t repetitionTable[1024];
static std::chrono::high_resolution_clock::time_point searchStart;
static int timeLimitMsGlobal;

bool outOfTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - searchStart).count();
    return elapsed >= timeLimitMsGlobal;
}


enum NodeType { EXACT, LOWERBOUND, UPPERBOUND };

struct TTEntry {
    uint64_t hash;
    int depth;
    int score;
    NodeType type;
    uint8_t age;
};

static constexpr int TT_SIZE = 1 << 22;
static TTEntry transTable[TT_SIZE];

inline TTEntry* probeTT(uint64_t hash)
{
    return &transTable[hash & (TT_SIZE - 1)];
}

inline void storeTT(uint64_t hash, int depth, int score, NodeType type)
{
    TTEntry* entry = probeTT(hash);

    if (entry->depth <= depth || entry->age != currentAge)
    {
        entry->hash = hash;
        entry->depth = depth;
        entry->score = score;
        entry->type  = type;
        entry->age   = currentAge;
    }
}

inline void clearTT()
{
    for (int i = 0; i < TT_SIZE; ++i)
    {
        transTable[i].depth = -1;
        transTable[i].hash  = 0;
    }
}
static chess::Move killerMoves[128][2];
static int historyHeuristic[2][64][64];

static const int knightPST[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
 };

static const int pawnPST[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10,-20,-20, 10, 10,  5,
    5, -5,-10,  0,  0,-10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
   10, 10, 20, 30, 30, 20, 10, 10,
   50, 50, 50, 50, 50, 50, 50, 50,
    0,  0,  0,  0,  0,  0,  0,  0
};

static const int bishopPST[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
 };
static const int rookPST[64] = {
    0,  0,  5, 10, 10,  5,  0,  0,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
    5, 10, 10, 10, 10, 10, 10,  5,
    0,  0,  5, 10, 10,  5,  0,  0
};
static const int queenPST[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};
static const int kingPST_MG[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
 };

static const int kingPST_EG[64] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
 };

int pieceValue(chess::Piece::underlying p) {
    switch (p) {
        case chess::Piece::underlying::WHITEPAWN:
        case chess::Piece::underlying::BLACKPAWN: return 100;
        case chess::Piece::underlying::WHITEKNIGHT:
        case chess::Piece::underlying::BLACKKNIGHT: return 320;
        case chess::Piece::underlying::WHITEBISHOP:
        case chess::Piece::underlying::BLACKBISHOP: return 330;
        case chess::Piece::underlying::WHITEROOK:
        case chess::Piece::underlying::BLACKROOK: return 500;
        case chess::Piece::underlying::WHITEQUEEN:
        case chess::Piece::underlying::BLACKQUEEN: return 900;
        default: return 0;
    }
}
int gamePhase(chess::Board& board)
{
    int phase = 0;

    for (int i = 0; i < 64; ++i)
    {
        auto piece = board.at(chess::Square(i)).internal();

        switch (piece)
        {
            case chess::Piece::underlying::WHITEKNIGHT:
            case chess::Piece::underlying::BLACKKNIGHT:
                phase += 1;
                break;

            case chess::Piece::underlying::WHITEBISHOP:
            case chess::Piece::underlying::BLACKBISHOP:
                phase += 1;
                break;

            case chess::Piece::underlying::WHITEROOK:
            case chess::Piece::underlying::BLACKROOK:
                phase += 2;
                break;

            case chess::Piece::underlying::WHITEQUEEN:
            case chess::Piece::underlying::BLACKQUEEN:
                phase += 4;
                break;

            default:
                break;
        }
    }

    return std::min(phase, 24);
}
int evaluateMaterial(chess::Board& board)
{
    int score = 0;

    for (int i = 0; i < 64; ++i)
    {
        auto piece = board.at(chess::Square(i)).internal();
        if (piece == chess::Piece::underlying::NONE) {
            continue;
        }

        int val = pieceValue(piece);

        if (chess::Piece(piece).color() == chess::Color::WHITE) {
            score += val;
        }
        else {
            score -= val;
        }
    }

    return score;
}
int evaluateMobility(chess::Board& board, int phase)
{
    if (phase < 6)
        return 0;

    int score = 0;

    for (int i = 0; i < 64; ++i)
    {
        auto piece = board.at(chess::Square(i)).internal();
        if (piece == chess::Piece::underlying::NONE)
            continue;

        bool isWhite = chess::Piece(piece).color() == chess::Color::WHITE;

        int file = i % 8;
        int rank = i / 8;

        int mobility = 0;

        switch (piece)
        {
            case chess::Piece::underlying::WHITEKNIGHT:
            case chess::Piece::underlying::BLACKKNIGHT:
            {
                static const int offsets[8][2] = {
                    {1,2},{2,1},{-1,2},{-2,1},
                    {1,-2},{2,-1},{-1,-2},{-2,-1}
                };

                for (auto& o : offsets)
                {
                    int f = file + o[0];
                    int r = rank + o[1];

                    if (f >= 0 && f < 8 && r >= 0 && r < 8)
                    {
                        auto target = board.at(chess::Square(r * 8 + f)).internal();
                        if (target == chess::Piece::underlying::NONE ||
                            chess::Piece(target).color() != chess::Piece(piece).color())
                        {
                            mobility++;
                        }
                    }
                }
                break;
            }

            case chess::Piece::underlying::WHITEBISHOP:
            case chess::Piece::underlying::BLACKBISHOP:
            case chess::Piece::underlying::WHITEROOK:
            case chess::Piece::underlying::BLACKROOK:
            case chess::Piece::underlying::WHITEQUEEN:
            case chess::Piece::underlying::BLACKQUEEN:
            {
                static const int directions[8][2] = {
                    {1,0},{-1,0},{0,1},{0,-1},
                    {1,1},{1,-1},{-1,1},{-1,-1}
                };

                int start = 0;
                int end = 8;

                if (piece == chess::Piece::underlying::WHITEBISHOP ||
                    piece == chess::Piece::underlying::BLACKBISHOP)
                {
                    start = 4;
                }
                else if (piece == chess::Piece::underlying::WHITEROOK ||
                         piece == chess::Piece::underlying::BLACKROOK)
                {
                    end = 4;
                }

                for (int d = start; d < end; ++d)
                {
                    int f = file;
                    int r = rank;

                    while (true)
                    {
                        f += directions[d][0];
                        r += directions[d][1];

                        if (f < 0 || f >= 8 || r < 0 || r >= 8)
                            break;

                        auto target = board.at(chess::Square(r * 8 + f)).internal();

                        if (target == chess::Piece::underlying::NONE)
                        {
                            mobility++;
                        }
                        else
                        {
                            if (chess::Piece(target).color() != chess::Piece(piece).color())
                                mobility++;
                            break;
                        }
                    }
                }
                break;
            }

            default:
                break;
        }

        int weight = (4 * phase + 1 * (24 - phase)) / 24;

        if (isWhite)
            score += mobility * weight;
        else
            score -= mobility * weight;
    }

    return score;
}
int evaluateKingSafety(chess::Board& board)
{
    int score = 0;

    for (int i = 0; i < 64; i++)
    {
        auto piece = board.at(chess::Square(i)).internal();
        if (piece == chess::Piece::underlying::WHITEKING)
        {
            int file = i % 8;
            int rank = i / 8;

            for (int df = -1; df <= 1; df++)
            {
                int f = file + df;
                int r = rank + 1;

                if (f >= 0 && f < 8 && r < 8)
                {
                    int sq = r * 8 + f;
                    if (board.at(chess::Square(sq)).internal() == chess::Piece::underlying::WHITEPAWN) {
                        score += 15;
                    }
                }
            }
        }

        if (piece == chess::Piece::underlying::BLACKKING)
        {
            int file = i % 8;
            int rank = i / 8;

            for (int df = -1; df <= 1; df++)
            {
                int f = file + df;
                int r = rank - 1;

                if (f >= 0 && f < 8 && r >= 0)
                {
                    int sq = r * 8 + f;
                    if (board.at(chess::Square(sq)).internal() == chess::Piece::underlying::BLACKPAWN) {
                        score -= 15;
                    }
                }
            }
        }
    }

    return score;
}
int evaluatePawnStructure(chess::Board& board, int phase)
{
    int score = 0;

    for (int file = 0; file < 8; file++)
    {
        int whiteCount = 0;
        int blackCount = 0;

        for (int rank = 0; rank < 8; rank++)
        {
            int sq = rank * 8 + file;
            auto piece = board.at(chess::Square(sq)).internal();

            if (piece == chess::Piece::underlying::WHITEPAWN) {
                whiteCount++;
            }
            if (piece == chess::Piece::underlying::BLACKPAWN) {
                blackCount++;
            }
        }

        if (whiteCount > 1) {
            score -= 15 * (whiteCount - 1);
        }

        if (blackCount > 1) {
            score += 15 * (blackCount - 1);
        }
    }

    for (int i = 0; i < 64; i++)
    {
        auto piece = board.at(chess::Square(i)).internal();
        if (piece == chess::Piece::underlying::WHITEPAWN)
        {
            int file = i % 8;
            int rank = i / 8;

            bool passed = true;

            for (int r = rank + 1; r < 8 && passed; r++)
            {
                for (int f = file - 1; f <= file + 1; f++)
                {
                    if (f >= 0 && f < 8)
                    {
                        int sq = r * 8 + f;
                        if (board.at(chess::Square(sq)).internal() ==  chess::Piece::underlying::BLACKPAWN)
                        {
                            passed = false;
                            break;
                        }
                    }
                }
            }

            if (passed) {
                int advancement = rank;
                int bonus = advancement * advancement * (phase < 12 ? 10 : 4);
                score += bonus;
            }
        }

        if (piece == chess::Piece::underlying::BLACKPAWN)
        {
            int file = i % 8;
            int rank = i / 8;

            bool passed = true;

            for (int r = rank - 1; r >= 0 && passed; r--)
            {
                for (int f = file - 1; f <= file + 1; f++)
                {
                    if (f >= 0 && f < 8)
                    {
                        int sq = r * 8 + f;
                        if (board.at(chess::Square(sq)).internal() == chess::Piece::underlying::WHITEPAWN)
                        {
                            passed = false;
                            break;
                        }
                    }
                }
            }

            if (passed) {
                int advancement = 7 - rank;
                int bonus = advancement * advancement * (phase < 12 ? 10 : 4);

                score -= bonus;
            }
        }
    }

    return score;
}
int evaluatePieceSquare(chess::Board& board, int phase)
{
    int score = 0;
    int whiteBishops = 0;
    int blackBishops = 0;

    for (int i = 0; i < 64; i++)
    {
        auto piece = board.at(chess::Square(i)).internal();
        if (piece == chess::Piece::underlying::NONE)
            continue;

        int sq = i;
        bool isWhite = chess::Piece(piece).color() == chess::Color::WHITE;

        int pstBonus = 0;

        switch (piece)
        {
            case chess::Piece::underlying::WHITEPAWN:
            case chess::Piece::underlying::BLACKPAWN:
                pstBonus = isWhite ? pawnPST[sq] : pawnPST[63 - sq];
                break;

            case chess::Piece::underlying::WHITEKNIGHT:
            case chess::Piece::underlying::BLACKKNIGHT:
                pstBonus = isWhite ? knightPST[sq] : knightPST[63 - sq];
                break;

            case chess::Piece::underlying::WHITEBISHOP:
            case chess::Piece::underlying::BLACKBISHOP:
                pstBonus = isWhite ? bishopPST[sq] : bishopPST[63 - sq];
                if (isWhite) whiteBishops++;
                else blackBishops++;
                break;

            case chess::Piece::underlying::WHITEROOK:
            case chess::Piece::underlying::BLACKROOK:
                pstBonus = isWhite ? rookPST[sq] : rookPST[63 - sq];
                break;

            case chess::Piece::underlying::WHITEQUEEN:
            case chess::Piece::underlying::BLACKQUEEN:
                pstBonus = isWhite ? queenPST[sq] : queenPST[63 - sq];
                break;
            case chess::Piece::underlying::WHITEKING:
            case chess::Piece::underlying::BLACKKING:
            {
                int mg = isWhite ? kingPST_MG[sq] : kingPST_MG[63 - sq];
                int eg = isWhite ? kingPST_EG[sq] : kingPST_EG[63 - sq];

                int kingScore = (mg * phase + eg * (24 - phase)) / 24;

                if (isWhite) {
                    score += kingScore;
                }
                else {
                    score -= kingScore;
                }
                break;
            }
            default:
                break;
        }

        if (isWhite)
            score += pstBonus;
        else
            score -= pstBonus;
    }

    if (whiteBishops >= 2) score += 30;
    if (blackBishops >= 2) score -= 30;

    return score;
}
int evaluateDevelopment(chess::Board& board)
{
    int score = 0;

    if (board.at(chess::Square::SQ_B1).internal() == chess::Piece::underlying::WHITEKNIGHT)
        score -= 15;
    if (board.at(chess::Square::SQ_G1).internal() == chess::Piece::underlying::WHITEKNIGHT)
        score -= 15;

    if (board.at(chess::Square::SQ_B8).internal() == chess::Piece::underlying::BLACKKNIGHT)
        score += 15;
    if (board.at(chess::Square::SQ_G8).internal() == chess::Piece::underlying::BLACKKNIGHT)
        score += 15;

    if (board.at(chess::Square::SQ_C1).internal() == chess::Piece::underlying::WHITEBISHOP)
        score -= 15;
    if (board.at(chess::Square::SQ_F1).internal() == chess::Piece::underlying::WHITEBISHOP)
        score -= 15;

    if (board.at(chess::Square::SQ_C8).internal() == chess::Piece::underlying::BLACKBISHOP)
        score += 15;
    if (board.at(chess::Square::SQ_F8).internal() == chess::Piece::underlying::BLACKBISHOP)
        score += 15;

    return score;
}
int evaluateCenterControl(chess::Board& board)
{
    int score = 0;

    int centerSquares[4] = { 27, 28, 35, 36 };

    for (int sq : centerSquares)
    {
        auto piece = board.at(chess::Square(sq)).internal();

        if (piece == chess::Piece::underlying::WHITEPAWN)
            score += 20;
        if (piece == chess::Piece::underlying::BLACKPAWN)
            score -= 20;
    }

    return score;
}
int evaluateRooks(chess::Board& board)
{
    int score = 0;

    for (int file = 0; file < 8; file++)
    {
        bool whitePawnOnFile = false;
        bool blackPawnOnFile = false;

        for (int rank = 0; rank < 8; rank++)
        {
            int sq = rank * 8 + file;
            auto piece = board.at(chess::Square(sq)).internal();

            if (piece == chess::Piece::underlying::WHITEPAWN)
                whitePawnOnFile = true;
            if (piece == chess::Piece::underlying::BLACKPAWN)
                blackPawnOnFile = true;
        }

        for (int rank = 0; rank < 8; rank++)
        {
            int sq = rank * 8 + file;
            auto piece = board.at(chess::Square(sq)).internal();

            if (piece == chess::Piece::underlying::WHITEROOK)
            {
                if (!whitePawnOnFile && !blackPawnOnFile)
                    score += 25;      // open file
                else if (!whitePawnOnFile)
                    score += 12;      // semi-open
            }

            if (piece == chess::Piece::underlying::BLACKROOK)
            {
                if (!whitePawnOnFile && !blackPawnOnFile)
                    score -= 25;
                else if (!blackPawnOnFile)
                    score -= 12;
            }
        }
    }

    return score;
}
int evaluateKingProx(chess::Board& board, int phase)
{

    if (phase < 8)
        return 0;

    int score = 0;

    int whiteKingSq = -1;
    int blackKingSq = -1;

    for (int i = 0; i < 64; i++)
    {
        auto piece = board.at(chess::Square(i)).internal();

        if (piece == chess::Piece::underlying::WHITEKING)
            whiteKingSq = i;

        if (piece == chess::Piece::underlying::BLACKKING)
            blackKingSq = i;
    }

    for (int i = 0; i < 64; i++)
    {
        auto piece = board.at(chess::Square(i)).internal();
        if (piece == chess::Piece::underlying::NONE)
            continue;

        chess::Color color = chess::Piece(piece).color();

        if (piece == chess::Piece::underlying::WHITEPAWN ||
            piece == chess::Piece::underlying::BLACKPAWN ||
            piece == chess::Piece::underlying::WHITEKING ||
            piece == chess::Piece::underlying::BLACKKING)
            continue;

        int targetSq = (color == chess::Color::WHITE) ? blackKingSq : whiteKingSq;

        int file = i % 8;
        int rank = i / 8;

        int tf = targetSq % 8;
        int tr = targetSq / 8;

        int distance = abs(file - tf) + abs(rank - tr);

        int baseBonus = std::max(0, 8 - distance);

        int pieceWeight = 0;

        switch (piece)
        {
            case chess::Piece::underlying::WHITEKNIGHT:
            case chess::Piece::underlying::BLACKKNIGHT:
                pieceWeight = 3;
                break;

            case chess::Piece::underlying::WHITEBISHOP:
            case chess::Piece::underlying::BLACKBISHOP:
                pieceWeight = 2;
                break;

            case chess::Piece::underlying::WHITEROOK:
            case chess::Piece::underlying::BLACKROOK:
                pieceWeight = 2;
                break;

            case chess::Piece::underlying::WHITEQUEEN:
            case chess::Piece::underlying::BLACKQUEEN:
                pieceWeight = 4;
                break;

            default:
                break;
        }

        int bonus = baseBonus * pieceWeight;

        // Taper by phase
        bonus = (bonus * phase) / 24;

        if (color == chess::Color::WHITE)
            score += bonus;
        else
            score -= bonus;
    }

    return score;
}
int evaluateKnightOutposts(chess::Board& board)
{
    int score = 0;

    for (int i = 0; i < 64; i++)
    {
        auto piece = board.at(chess::Square(i)).internal();

        if (piece != chess::Piece::underlying::WHITEKNIGHT &&
            piece != chess::Piece::underlying::BLACKKNIGHT)
            continue;

        bool isWhite = chess::Piece(piece).color() == chess::Color::WHITE;
        int file = i % 8;
        int rank = i / 8;

        bool supported = false;

        if (isWhite && rank > 0)
        {
            for (int f = file - 1; f <= file + 1; f += 2)
            {
                if (f >= 0 && f < 8)
                {
                    int sq = (rank - 1) * 8 + f;
                    if (board.at(chess::Square(sq)).internal() ==
                        chess::Piece::underlying::WHITEPAWN)
                        supported = true;
                }
            }
        }

        if (!isWhite && rank < 7)
        {
            for (int f = file - 1; f <= file + 1; f += 2)
            {
                if (f >= 0 && f < 8)
                {
                    int sq = (rank + 1) * 8 + f;
                    if (board.at(chess::Square(sq)).internal() ==
                        chess::Piece::underlying::BLACKPAWN)
                        supported = true;
                }
            }
        }

        if (supported)
        {
            if (isWhite)
                score += 25;
            else
                score -= 25;
        }
    }

    return score;
}
int evaluateKingActivity(chess::Board& board, int phase)
{
    int score = 0;

    if (phase > 8)
        return 0;

    for (int i = 0; i < 64; ++i)
    {
        auto piece = board.at(chess::Square(i)).internal();

        if (piece == chess::Piece::underlying::WHITEKING ||
            piece == chess::Piece::underlying::BLACKKING)
        {
            int file = i % 8;
            int rank = i / 8;

            int distFromCenter = abs(file - 3) + abs(rank - 3);

            int bonus = 14 - distFromCenter * 2;

            if (chess::Piece(piece).color() == chess::Color::WHITE)
                score += bonus;
            else
                score -= bonus;
        }
    }

    return score;
}

int evaluate(chess::Board& board)
{
    int phase = gamePhase(board);

    int material = evaluateMaterial(board);
    int pawnStructure = evaluatePawnStructure(board, phase);
    int mobility = evaluateMobility(board, phase);
    int pieceSquare = evaluatePieceSquare(board, phase);
    int rooks = evaluateRooks(board);
    int kingProxim = evaluateKingProx(board, phase);
    int kingActive = evaluateKingActivity(board, phase);

    int mgScore = 0;
    int egScore = 0;

    mgScore += material;
    mgScore += pawnStructure;
    mgScore += mobility;
    mgScore += evaluateKingSafety(board);
    mgScore += pieceSquare;
    mgScore += evaluateDevelopment(board);
    mgScore += evaluateCenterControl(board);
    mgScore += rooks;
    mgScore += kingProxim;
    mgScore += evaluateKnightOutposts(board);

    egScore += material;
    egScore += pawnStructure;
    //egScore += mobility;
    egScore += pieceSquare;
    egScore += rooks;
    egScore += kingActive;

    int score = (mgScore * phase + egScore * (24 - phase)) / 24;

    return (board.sideToMove() == chess::Color::WHITE) ? score : -score;
}


int captureScore(chess::Board& board, chess::Move move) {
    auto victim = board.at(move.to()).internal();
    auto attacker = board.at(move.from()).internal();
    return pieceValue(victim) - pieceValue(attacker);
}

int scoreMove(chess::Board& board, chess::Move move, int depth) {
    if (board.isCapture(move))
        return 100000 + captureScore(board, move);

    if (move == killerMoves[depth][0])
        return 90000;

    if (move == killerMoves[depth][1])
        return 80000;

    int side = (board.sideToMove() == chess::Color::WHITE) ? 0 : 1;
    return historyHeuristic[side][move.from().index()][move.to().index()];
}

void orderMoves(chess::Movelist& moves, chess::Board& board, int depth) {
    std::sort(moves.begin(), moves.end(),
        [&](const chess::Move& a, const chess::Move& b) {
            return scoreMove(board, a, depth) >
                   scoreMove(board, b, depth);
        });
}

int quiescence(chess::Board& board, int alpha, int beta) {
    int standPat = evaluate(board);

    if (standPat >= beta)
        return beta;

    if (alpha < standPat)
        alpha = standPat;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    for (auto move : moves) {
        if (!board.isCapture(move))
            continue;

        board.makeMove(move);
        int score = -quiescence(board, -beta, -alpha);
        board.unmakeMove(move);

        if (score >= beta)
            return beta;

        if (score > alpha)
            alpha = score;
    }

    return alpha;
}

int alphaBeta(chess::Board& board, int depth, int alpha, int beta, int ply)
{
    if (outOfTime()) {
        return evaluate(board);
    }
    uint64_t hash = board.hash();

    int repetitionCount = 0;

    for (int i = 0; i < ply; ++i)
    {
        if (repetitionTable[i] == hash)
        {
            repetitionCount++;
            if (repetitionCount >= 2)
            {
                return 0;
            }
        }
    }

    if (depth == 0) {
        return quiescence(board, alpha, beta);
    }

    auto gameOver = board.isGameOver();
    if (gameOver.second != chess::GameResult::NONE)
    {
        if (gameOver.second == chess::GameResult::DRAW) {
            return 0;
        }
        return -MATE_SCORE + ply;
    }

    TTEntry* tt = probeTT(hash);

    if (tt->hash == hash && tt->depth >= depth)
    {
        if (tt->type == EXACT)
        {
            return tt->score;
        }
        else if (tt->type == LOWERBOUND)
        {
            alpha = std::max(alpha, tt->score);
        }
        else if (tt->type == UPPERBOUND) {
            beta = std::min(beta, tt->score);
        }
        if (alpha >= beta){
            return tt->score;
        }
    }
    if (depth == 1 && !board.inCheck())
    {
        int staticEval = evaluate(board);
        if (staticEval + 100 <= alpha)
            return staticEval;
    }

    if (depth >= 3 && !board.inCheck() && gamePhase(board) > 6)
    {
        int R = (depth >= 6) ? 3 : 2;

        board.makeNullMove();
        int score = -alphaBeta(board, depth - 1 - R, -beta, -beta + 1, ply + 1);
        board.unmakeNullMove();

        if (score >= beta) {
            return beta;
        }
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    orderMoves(moves, board, ply);

    int originalAlpha = alpha;
    int bestScore = -INF;
    bool firstMove = true;
    int moveCount = 0;

    for (auto move : moves)
    {
        moveCount++;
        int score;
        bool inCheck = board.inCheck();
        bool isCapture = board.isCapture(move);

        if (depth <= 3 &&  moveCount > 6 && !isCapture && !inCheck)
        {
            continue;
        }
        repetitionTable[ply] = hash;
        board.makeMove(move);

        if (firstMove)
        {
            score = -alphaBeta(board, depth - 1, -beta, -alpha, ply + 1);
            firstMove = false;
        }
        else
        {
            int reduction = 0;

            if (depth >= 3 &&
                moveCount > 1 &&
                !isCapture &&
                !inCheck)
            {
                reduction = (moveCount > 6) ? 2 : 1;
            }

            score = -alphaBeta(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);

            if (score > alpha)
                score = -alphaBeta(board, depth - 1, -beta, -alpha, ply + 1);
        }

        board.unmakeMove(move);
        repetitionTable[ply] = 0;

        if (score > bestScore) {
            bestScore = score;
        }

        if (score > alpha)
        {
            alpha = score;

            if (!board.isCapture(move))
            {
                killerMoves[ply][1] = killerMoves[ply][0];
                killerMoves[ply][0] = move;

                int side = (board.sideToMove() == chess::Color::WHITE) ? 0 : 1;
                historyHeuristic[side][move.from().index()][move.to().index()] += depth * depth;
            }
        }

        if (alpha >= beta)
            break;
    }
    if (moves.empty())
    {
        if (board.inCheck()) {
            return -MATE_SCORE + ply;
        }
        else {
            return 0;
        }
    }
    NodeType type;
    if (bestScore <= originalAlpha)
        type = UPPERBOUND;
    else if (bestScore >= beta)
        type = LOWERBOUND;
    else
        type = EXACT;

    storeTT(hash, depth, bestScore, type);

    return bestScore;
}

std::string ChessSimulator::Move(std::string fen, int timeLimitMs)
{
    chess::Board board(fen);

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty())
        return "";

    searchStart = std::chrono::high_resolution_clock::now();
    timeLimitMsGlobal = timeLimitMs;

    clearTT();
    currentAge++;

    chess::Move bestMove = moves[0];
    chess::Move pvMove;
    int bestScore = 0;

    for (int depth = 1; depth <= MAX_DEPTH; depth++)
    {
        if (outOfTime())
            break;

        orderMoves(moves, board, 0);

        if (pvMove != chess::Move())
        {
            auto it = std::find(moves.begin(), moves.end(), pvMove);
            if (it != moves.end())
                std::iter_swap(moves.begin(), it);
        }

        int window = 50;
        int alpha, beta;

        if (depth == 1)
        {
            alpha = -INF;
            beta  = INF;
        }
        else
        {
            alpha = bestScore - window;
            beta  = bestScore + window;
        }

        while (true)
        {
            int localBestScore = -INF;
            chess::Move localBestMove = bestMove;

            int a = alpha;
            int b = beta;
            bool firstMove = true;

            for (auto move : moves)
            {
                board.makeMove(move);

                int moveScore;

                if (firstMove)
                {
                    moveScore = -alphaBeta(board, depth - 1, -b, -a, 1);
                    firstMove = false;
                }
                else
                {
                    moveScore = -alphaBeta(board, depth - 1, -a - 1, -a, 1);

                    if (moveScore > a)
                        moveScore = -alphaBeta(board, depth - 1, -b, -a, 1);
                }

                board.unmakeMove(move);

                if (outOfTime())
                    break;

                if (moveScore > localBestScore)
                {
                    localBestScore = moveScore;
                    localBestMove  = move;
                }

                if (moveScore > a)
                    a = moveScore;
            }

            if (outOfTime())
                break;

            int score = localBestScore;

            if (score <= alpha)
            {
                alpha -= window;
                window *= 2;
            }
            else if (score >= beta)
            {
                beta += window;
                window *= 2;
            }
            else
            {
                bestScore = score;
                bestMove  = localBestMove;
                pvMove    = bestMove;
                break;
            }
        }
    }

    return chess::uci::moveToUci(bestMove);
}
