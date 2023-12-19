/*
 * Copyright 2023 Alexey Odinokov. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.aodinokov.setgameresolver

enum class CardNumber(val code: Int) {
    ONE(1),
    TWO(2),
    THREE(3);
    override fun toString(): String {
        return code.toString()
    }
    companion object {
        fun fromInt(value: Int):CardNumber? {
            return try {
                CardNumber.values().first { it.code == value }
            }catch (ex: NoSuchElementException) {
                null
            }
        }
        fun fromString(value: String):CardNumber? {
            return try {
                CardNumber.values().first { it.toString().lowercase() == value.lowercase() }
            }catch (ex: NoSuchElementException) {
                null
            }
        }
        fun previous(current: CardNumber): CardNumber {return CardNumber.fromInt((((current.code - 1) + 2) % 3) + 1)!!}
        fun next(current: CardNumber): CardNumber {return CardNumber.fromInt((((current.code - 1) + 1) % 3) + 1)!!}
    }
}

enum class CardColor(val code: Int) {
    GREEN(1),
    PURPLE(2),
    RED(3);
    override fun toString(): String {
        return super.toString().lowercase()
    }
    companion object {
        fun fromInt(value: Int):CardColor? {
            return try {
                CardColor.values().first { it.code == value }
            }catch (ex: NoSuchElementException) {
                null
            }
        }
        fun fromString(value: String):CardColor? {
            return try {
                CardColor.values().first { it.toString().lowercase() == value.lowercase() }
            }catch (ex: NoSuchElementException) {
                null
            }
        }
        fun previous(current: CardColor): CardColor {return CardColor.fromInt((((current.code - 1) + 2) % 3) + 1)!!}
        fun next(current: CardColor): CardColor {return CardColor.fromInt((((current.code - 1) + 1) % 3) + 1)!!}
    }
}

enum class CardShading(val code: Int) {
    EMPTY(1),
    SOLID(2),
    STRIPED(3); // or Outlined
    override fun toString(): String {
        return super.toString().lowercase()
    }
    companion object {
        fun fromInt(value: Int):CardShading? {
            return try {
                CardShading.values().first { it.code == value }
            }catch (ex: NoSuchElementException) {
                null
            }
        }
        fun fromString(value: String):CardShading? {
            return try {
                CardShading.values().first { it.toString().lowercase() == value.lowercase() }
            }catch (ex: NoSuchElementException) {
                null
            }
        }
        fun previous(current: CardShading): CardShading {return CardShading.fromInt((((current.code - 1) + 2) % 3) + 1)!!}
        fun next(current: CardShading): CardShading {return CardShading.fromInt((((current.code - 1) + 1) % 3) + 1)!!}
    }
}

enum class CardShape(val code: Int) {
    DIAMOND(1),
    OVAL(2),
    SQUIGGLE(3),;
    override fun toString(): String {
        return super.toString().lowercase()
    }
    companion object {
        fun fromInt(value: Int):CardShape? {
            return try {
                CardShape.values().first { it.code == value }
            }catch (ex: NoSuchElementException) {
                null
            }
        }
        fun fromString(value: String):CardShape? {
            return try {
                CardShape.values().first { it.toString().lowercase() == value.lowercase() }
            }catch (ex: NoSuchElementException) {
                null
            }
        }
        fun previous(current: CardShape): CardShape {return CardShape.fromInt((((current.code - 1) + 2) % 3) + 1)!!}
        fun next(current: CardShape): CardShape {return CardShape.fromInt((((current.code - 1) + 1) % 3) + 1)!!}
    }

}
data class CardValue(val number: CardNumber, var color: CardColor, var shading: CardShading, var shape: CardShape){
    override fun toString(): String {
        val res = String.format("%s-%s-%s-%s",
                number.toString(),
                color.toString(),
                shading.toString(),
                shape.toString())
        if (number != CardNumber.ONE) {
            // make plural
            return res + "s"
        }
        return res
    }
    companion object {
        fun fromString(value: String):CardValue? {
            val parts = value.split("-", ignoreCase=true, limit = 4)
            if (parts.size != 4) {
                return null
            }
            val number = CardNumber.fromString(parts.elementAt(0)) ?: return null
            val color = CardColor.fromString(parts.elementAt(1)) ?: return null
            val shading = CardShading.fromString(parts.elementAt(2)) ?: return null
            if (/*number != CardNumber.ONE &&*/ parts.elementAt(3).last() == 's') {
                // support plural
                val shape = CardShape.fromString(parts.elementAt(3).dropLast(1)) ?: return null
                return CardValue(number, color, shading, shape)
            }
            val shape = CardShape.fromString(parts.elementAt(3)) ?: return null
            return CardValue(number, color, shading, shape)
        }
    }
}

abstract class AbstractCard {
    abstract fun getValue(): CardValue
    override fun toString(): String {
        return getValue().toString()
    }
}

open class SimpleCard(private val v: CardValue): AbstractCard() {
    override fun getValue(): CardValue {
        return v
    }
}

// Note: SET is a CardSet of 3 cards in accordance with the rules in our model
class CardSet: HashSet<AbstractCard> {
    constructor():super()
    constructor(collection: Collection<AbstractCard>):super(collection)

    override fun toString(): String {
        // sort them within the set
        // so we could compare string names to compare combinations
        val sorted = this.sortedBy {
            it.getValue().number.code*27 +
                    it.getValue().color.code * 9 +
                    it.getValue().shading.code * 3 +
                    it.getValue().shape.code }
        var res = ""
        for (i in sorted) {
            if (res.isNotEmpty()){
                res += ", "
            }
            res += i.toString()
        }
        return "($res)"
    }
    companion object {
        private fun overlap(s1: CardSet, s2: CardSet): Boolean {
            for (c in s1) {
                if (s2.contains(c)) {
                    return true
                }
            }
            return false
        }
        /**
         * Convert the input set of card into list of sets 3 cards per each
         * based on the rules of the game
         */
        fun findAllSets(input: Set<AbstractCard>): Set<CardSet> {
            val result = HashSet<CardSet>()
            for(i in input.indices) {
                val c0 = input.elementAt(i)
                for (j in i+1 until input.size) {
                    val c1 = input.elementAt(j)
                    for (k in j+1 until input.size) {
                        val c2 = input.elementAt(k)

                        val count = c0.getValue().number.code + c1.getValue().number.code + c2.getValue().number.code
                        val color = c0.getValue().color.code + c1.getValue().color.code + c2.getValue().color.code
                        val shading = c0.getValue().shading.code + c1.getValue().shading.code + c2.getValue().shading.code
                        val shape = c0.getValue().shape.code + c1.getValue().shape.code +c2.getValue().shape.code

                        if (    count%3 == 0 &&
                                color%3 == 0 &&
                                shading%3 == 0 &&
                                shape%3 == 0) {
                            result.add(CardSet(setOf(c0, c1, c2)))
                        }
                    }
                }
            }
            return result
        }
        /**
         *  findAllSolutions can return Solutions that use the same (overlapping) cards
         *  findAllNonOverlappingSets tries to find a combination of sets with maximum number of sets.
         *  If there are several such combinations - it returns all (that's why
         *  it returns the outer set that contains N same length sets of set of 3 card in each)
         */
        fun findAllNonOverlappingSets(input: Set<CardSet>): Set<Set<CardSet>> {
            for (i in input.indices) {
                for (j in i+1 until input.size) {
                    if (overlap(input.elementAt(i), input.elementAt(j))) {
                        // build 2 subsets and check them separately
                        val res1: Set<Set<CardSet>> =
                                findAllNonOverlappingSets(input.minusElement(input.elementAt(j)))
                        val res2: Set<Set<CardSet>> =
                                findAllNonOverlappingSets(input.minusElement(input.elementAt(i)))

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
            return setOf(input)
        }
    }
}
