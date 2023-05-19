package org.tensorflow.lite.examples.objectdetection

fun countFromString(input: String): Int? {
    if (input == "1")
        return 1
    if (input == "2")
        return 2
    if (input == "3")
        return 3
    return null
}

enum class CardColor(val code: Int) {
    RED(1),
    PURPLE(2),
    GREEN(3),
}

fun colorFromString(input: String): CardColor? {
    if (input == "red")
        return CardColor.RED
    if (input == "green")
        return CardColor.GREEN
    if (input == "purple")
        return CardColor.PURPLE
    return null
}

fun colorToString(input: CardColor): String {
    if (input == CardColor.RED)
        return "red"
    if (input == CardColor.GREEN)
        return "green"
    if (input == CardColor.PURPLE)
        return "purple"
    return ""
}


enum class CardFill(val code: Int) {
    SOLID(1),
    STRIPED(2),
    EMPTY(3),
}

fun fillFromString(input: String): CardFill? {
    if (input == "empty")
        return CardFill.EMPTY
    if (input == "striped")
        return CardFill.STRIPED
    if (input == "solid")
        return CardFill.SOLID
    return null
}

fun fillToString(input: CardFill): String {
    if (input == CardFill.EMPTY)
        return "empty"
    if (input == CardFill.STRIPED)
        return "striped"
    if (input == CardFill.SOLID)
        return "solid"
    return ""
}

enum class CardShape(val code: Int) {
    SQUIGGLE(1),
    DIAMOND(2),
    OVAL(3),
}

fun shapeFromString(input: String): CardShape? {
    if (    input == "oval" ||
            input == "ovals")
        return CardShape.OVAL
    if (    input == "diamond" ||
            input == "diamonds")
        return CardShape.DIAMOND
    if (    input == "squiggle" ||
            input == "squiggles")
        return CardShape.SQUIGGLE
    return null
}

fun shapeFromString(n: Int, input: CardShape): String {
    if (n>1) {
        if (input == CardShape.OVAL)return "oval"
        if (input == CardShape.DIAMOND)return "diamond"
        if (input == CardShape.SQUIGGLE)return "squiggle"
    } else {
        if (input == CardShape.OVAL)return "ovals"
        if (input == CardShape.DIAMOND)return "diamonds"
        if (input == CardShape.SQUIGGLE)return "squiggles"
    }
    return ""
}

data class Card(val count: Int, var cardColor: CardColor, var cardFill: CardFill, var cardShape: CardShape)
data class Solution(var cards: Set<Card>)

fun cardFromString(input: String): Card? {
    val parts = input.split("-", ignoreCase=true, limit = 4)
    if (parts.size != 4) {
        return null
    }
    val count: Int = countFromString(parts.elementAt(0)) ?: return null
    val cardColor: CardColor = colorFromString(parts.elementAt(1)) ?: return null
    val cardFill: CardFill = fillFromString(parts.elementAt(2)) ?: return null
    val cardShape: CardShape = shapeFromString(parts.elementAt(3)) ?: return null
    return Card(count, cardColor, cardFill, cardShape)
}

fun cardToString(crd: Card): String {
    return String.format("%d-%s-%s-%s",
            crd.count,
            colorToString(crd.cardColor),
            fillToString(crd.cardFill),
            shapeFromString(crd.count, crd.cardShape)
            )
}

/**
 * Convert the input set of card into list of sets 3 cards per each
 * based on the rules of the game
 */
fun findAllSolutions(input: Set<Card>): Set<Solution> {
    val result = HashSet<Solution>()
    for(i in 0..(input.size-1)) {
        val c0 = input.elementAt(i)
        for (j in i+1 .. (input.size-1)) {
            val c1 = input.elementAt(j)
            for (k in j+1 .. (input.size-1)) {
                val c2 = input.elementAt(k)

                val count = c0.count + c1.count + c2.count
                val color = c0.cardColor.code + c1.cardColor.code + c2.cardColor.code
                val fill = c0.cardFill.code + c1.cardFill.code + c2.cardFill.code
                val shape = c0.cardShape.code + c1.cardShape.code +c2.cardShape.code

                if (    count%3 == 0 &&
                        color%3 == 0 &&
                        fill%3 == 0 &&
                        shape%3 == 0) {
                    val solution = Solution(cards = setOf(c0, c1, c2))
                    result.add(solution)
                }
            }
        }
    }
    return result
}

/**
 *  FindAllSolutions can return Solutions that use the same (overlapping) cards
 *  This function tries to find a combination of sets with maximum number of sets.
 *  If there are several such combinations - it returns all (that's why it returns the list)
 */
fun findAllNonOverlappingSolutions(input: Set<Solution>): List<Set<Solution>> {
    for (i in 0..(input.size-1)) {
        for (j in i+1 .. (input.size-1)) {
            if (areSolutionsOverlap(input.elementAt(i), input.elementAt(j))) {
                // build 2 subsets and check them separately
                val res1: List<Set<Solution>> =
                    findAllNonOverlappingSolutions(input.minus(input.elementAt(j)))
                val res2: List<Set<Solution>> =
                    findAllNonOverlappingSolutions(input.minus(input.elementAt(i)))

                // we assume that result has at least 1 element. This must be always true
                assert(res1.isNotEmpty())
                assert(res2.isNotEmpty())

                if (res1.elementAt(0).size == res2.elementAt(0).size) {
                    // we have got a same number of sets in both cases, they are all interchangeable
                    // we can merge those solutions and choose any of them
                    return res1 + res2
                }
                if (res1.elementAt(0).size > res2.elementAt(0).size) {
                    // the solutions in res1 have more sets
                    return res1
                }
                return res2
            }
        }
    }
    // we didn't find any overlaps - we have a ready solution
    return listOf(input)
}

fun areSolutionsOverlap(s1: Solution, s2: Solution): Boolean {
    for (c in s1.cards) {
        if (s2.cards.contains(c)) {
            return true
        }
    }
    return false
}