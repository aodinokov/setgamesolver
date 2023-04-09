package org.tensorflow.lite.examples.objectdetection

import java.util.EmptyStackException

fun countFromString(input: String): Int? {
    if (input == "1")
        return 1
    if (input == "2")
        return 2
    if (input == "3")
        return 3
    return null
}

enum class Color(val code: Int) {
    RED(1),
    GREEN(2),
    PURPLE(3)
}

fun colorFromString(input: String): Color? {
    if (input == "red")
        return Color.RED
    if (input == "green")
        return Color.GREEN
    if (input == "purple")
        return Color.PURPLE
    return null
}

enum class Fill(val code: Int) {
    EMPTY(1),
    STRIPED(2),
    SOLID(3)
}

fun fillFromString(input: String): Fill? {
    if (input == "empty")
        return Fill.EMPTY
    if (input == "striped")
        return Fill.STRIPED
    if (input == "solid")
        return Fill.SOLID
    return null
}

enum class Shape(val code: Int) {
    OVAL(1),
    DIAMOND(2),
    SQUIGGLE(3)
}

fun shapeFromString(input: String): Shape? {
    if (    input == "oval" ||
            input == "ovals")
        return Shape.OVAL
    if (    input == "diamond" ||
            input == "diamonds")
        return Shape.DIAMOND
    if (    input == "squiggle" ||
            input == "squiggles")
        return Shape.SQUIGGLE
    return null
}

data class Card(val count: Int, var color: Color, var fill: Fill, var shape: Shape)
data class Solution(var cards: Set<Card>)

fun cardFromString(input: String): Card? {
    var parts = input.split("-", ignoreCase=true, limit = 4)
    if (parts.size != 4) {
        return null
    }

    var count = countFromString(parts.elementAt(0))
    if (count == null)
        return null

    var color = colorFromString(parts.elementAt(1))
    if (color == null)
        return null

    var fill = fillFromString(parts.elementAt(2))
    if (fill == null)
        return null

    return Card(count, color, Fill.EMPTY, Shape.DIAMOND)
}

/**
 * Convert the input set of card into list of sets 3 cards per each
 * based on the rules of the game
 */
fun findAllSolutions(input: Set<Card>): Set<Solution> {
    val result: Set<Solution> = emptySet()
    for(i in 0..(input.size-1)) {
        val c0 = input.elementAt(i)
        for (j in i+1 .. (input.size-1)) {
            val c1 = input.elementAt(j)
            for (k in j+1 .. (input.size-1)) {
                val c2 = input.elementAt(k)

                val count = c0.count + c1.count + c2.count
                val color = c0.color.code + c1.color.code + c2.color.code
                val fill = c0.fill.code + c1.fill.code + c2.fill.code
                val shape = c0.shape.code + c1.shape.code +c2.shape.code

                if (    count%3 == 0 &&
                        color%3 == 0 &&
                        fill%3 == 0 &&
                        shape%3 == 0) {
                    val solution = Solution(cards = setOf(c0, c1, c2))
                    result.plus(solution)
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
fun findAllNonOverlappingSulutions(input: Set<Solution>): List<Set<Solution>> {
    for (i in 0..(input.size-1)) {
        for (j in i+1 .. (input.size-1)) {
            if (areSolutionsOverlap(input.elementAt(i), input.elementAt(j))) {
                // build 2 subsets and check them separately
                var res1: List<Set<Solution>> =
                    findAllNonOverlappingSulutions(input.minus(input.elementAt(j)))
                var res2: List<Set<Solution>> =
                    findAllNonOverlappingSulutions(input.minus(input.elementAt(i)))

                // we assume that result has at least 1 element. This must be always true
                assert(res1.isNotEmpty())
                assert(res2.isNotEmpty())

                if (res1.elementAt(0).size == res2.elementAt(0).size) {
                    // we have got a same number of sets in both cases - they are all interchangable
                    // we can merge those solutions and choose any of them
                    return res1 + res2
                }
                if (res1.elementAt(0).size > res2.elementAt(0).size) {
                    // the solutions in res1 have more sets
                    return res1
                } else {
                    return res2
                }
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