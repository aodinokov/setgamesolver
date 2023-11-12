package org.tensorflow.lite.examples.objectdetection

fun cardNumberFromString(input: String): Int? {
    val res = input.toInt()
    if (res in 1..3)
        return res
    return null
}

fun cardNumberMinus(current: Int): Int {
    return (((current - 1) + 2) % 3) + 1
}
fun cardNumberPlus(current: Int): Int {
    return (((current - 1) + 1) % 3) + 1
}

enum class CardColor(val code: Int) {
    RED(1),
    PURPLE(2),
    GREEN(3);
    companion object {
        fun fromInt(value: Int) = CardColor.values().first { it.code == value}
    }
}

fun cardColorMinus(current: CardColor): CardColor {
    return CardColor.fromInt((((current.code - 1) + 2) % 3) + 1)
}
fun cardColorPlus(current: CardColor): CardColor {
    return CardColor.fromInt((((current.code - 1) + 1) % 3) + 1)
}

fun cardColorFromString(input: String): CardColor? {
    if (input == "red")
        return CardColor.RED
    if (input == "green")
        return CardColor.GREEN
    if (input == "purple")
        return CardColor.PURPLE
    return null
}

fun cardColorToString(input: CardColor): String {
    if (input == CardColor.RED)
        return "red"
    if (input == CardColor.GREEN)
        return "green"
    if (input == CardColor.PURPLE)
        return "purple"
    return ""
}

enum class CardShading(val code: Int) {
    SOLID(1),
    STRIPED(2),
    EMPTY(3); // or Outlined
    companion object {
        fun fromInt(value: Int) = CardShading.values().first { it.code == value}
    }
}

fun cardShadingMinus(current: CardShading): CardShading {
    return CardShading.fromInt((((current.code - 1) + 2) % 3) + 1)
}
fun cardShadingPlus(current: CardShading): CardShading {
    return CardShading.fromInt((((current.code - 1) + 1) % 3) + 1)
}

fun cardShadingFromString(input: String): CardShading? {
    if (input == "empty")
        return CardShading.EMPTY
    if (input == "striped")
        return CardShading.STRIPED
    if (input == "solid")
        return CardShading.SOLID
    return null
}

fun cardShadingToString(input: CardShading): String {
    if (input == CardShading.EMPTY)
        return "empty"
    if (input == CardShading.STRIPED)
        return "striped"
    if (input == CardShading.SOLID)
        return "solid"
    return ""
}

enum class CardShape(val code: Int) {
    SQUIGGLE(1),
    DIAMOND(2),
    OVAL(3);
    companion object {
        fun fromInt(value: Int) = CardShape.values().first { it.code == value}
    }
}

fun cardShapeMinus(current: CardShape): CardShape {
    return CardShape.fromInt((((current.code - 1) + 2) % 3) + 1)
}
fun cardShapePlus(current: CardShape): CardShape {
    return CardShape.fromInt((((current.code - 1) + 1) % 3) + 1)
}

fun cardShapeFromString(input: String): CardShape? {
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

fun cardShapeToString(n: Int, input: CardShape): String {
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

data class CardValue(val number: Int, var color: CardColor, var shading: CardShading, var shape: CardShape)
fun cardValueToString(cardValue: CardValue): String {
    return String.format("%d-%s-%s-%s",
            cardValue.number,
            cardColorToString(cardValue.color),
            cardShadingToString(cardValue.shading),
            cardShapeToString(cardValue.number, cardValue.shape)
    )
}

fun cardValueFromString(input: String): CardValue? {
    val parts = input.split("-", ignoreCase=true, limit = 4)
    if (parts.size != 4) {
        return null
    }
    val count: Int = cardNumberFromString(parts.elementAt(0)) ?: return null
    val cardColor: CardColor = cardColorFromString(parts.elementAt(1)) ?: return null
    val cardShading: CardShading = cardShadingFromString(parts.elementAt(2)) ?: return null
    val cardShape: CardShape = cardShapeFromString(parts.elementAt(3)) ?: return null
    return CardValue(count, cardColor, cardShading, cardShape)
}

// Note: we call SETS as SetCombination in our model
data class SetCombination(var cardValues: Set<CardValue>)

/**
 * Convert the input set of card into list of sets 3 cards per each
 * based on the rules of the game
 */
fun findAllSetCombination(input: Set<CardValue>): Set<SetCombination> {
    val result = HashSet<SetCombination>()
    for(i in input.indices) {
        val c0 = input.elementAt(i)
        for (j in i+1 until input.size) {
            val c1 = input.elementAt(j)
            for (k in j+1 until input.size) {
                val c2 = input.elementAt(k)

                val count = c0.number + c1.number + c2.number
                val color = c0.color.code + c1.color.code + c2.color.code
                val shading = c0.shading.code + c1.shading.code + c2.shading.code
                val shape = c0.shape.code + c1.shape.code +c2.shape.code

                if (    count%3 == 0 &&
                        color%3 == 0 &&
                        shading%3 == 0 &&
                        shape%3 == 0) {
                    val setCombination = SetCombination(cardValues = setOf(c0, c1, c2))
                    result.add(setCombination)
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
fun findAllNonOverlappingSetCombination(input: Set<SetCombination>): List<Set<SetCombination>> {
    for (i in input.indices) {
        for (j in i+1 until input.size) {
            if (areSolutionsOverlap(input.elementAt(i), input.elementAt(j))) {
                // build 2 subsets and check them separately
                val res1: List<Set<SetCombination>> =
                    findAllNonOverlappingSetCombination(input.minus(input.elementAt(j)))
                val res2: List<Set<SetCombination>> =
                    findAllNonOverlappingSetCombination(input.minus(input.elementAt(i)))

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
    // we didn't find any overlaps - we have a ready SET
    return listOf(input)
}

fun areSolutionsOverlap(s1: SetCombination, s2: SetCombination): Boolean {
    for (c in s1.cardValues) {
        if (s2.cardValues.contains(c)) {
            return true
        }
    }
    return false
}