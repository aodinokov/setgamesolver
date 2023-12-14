/*
 * Copyright 2022 Alexey Odinokov. All Rights Reserved.
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

package org.tensorflow.lite.examples.objectdetection

import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Local tests
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class SetgameObjectModelTest {

    // taken from the classification model
    private val validLabels = listOf<String>(
            "1-green-empty-diamond",
            "1-green-empty-oval",
            "1-green-empty-squiggle",
            "1-green-solid-diamond",
            "1-green-solid-oval",
            "1-green-solid-squiggle",
            "1-green-striped-diamond",
            "1-green-striped-oval",
            "1-green-striped-squiggle",
            "1-purple-empty-diamond",
            "1-purple-empty-oval",
            "1-purple-empty-squiggle",
            "1-purple-solid-diamond",
            "1-purple-solid-oval",
            "1-purple-solid-squiggle",
            "1-purple-striped-diamond",
            "1-purple-striped-oval",
            "1-purple-striped-squiggle",
            "1-red-empty-diamond",
            "1-red-empty-oval",
            "1-red-empty-squiggle",
            "1-red-solid-diamond",
            "1-red-solid-oval",
            "1-red-solid-squiggle",
            "1-red-striped-diamond",
            "1-red-striped-oval",
            "1-red-striped-squiggle",
            "2-green-empty-diamonds",
            "2-green-empty-ovals",
            "2-green-empty-squiggles",
            "2-green-solid-diamonds",
            "2-green-solid-ovals",
            "2-green-solid-squiggles",
            "2-green-striped-diamonds",
            "2-green-striped-ovals",
            "2-green-striped-squiggles",
            "2-purple-empty-diamonds",
            "2-purple-empty-ovals",
            "2-purple-empty-squiggles",
            "2-purple-solid-diamonds",
            "2-purple-solid-ovals",
            "2-purple-solid-squiggles",
            "2-purple-striped-diamonds",
            "2-purple-striped-ovals",
            "2-purple-striped-squiggles",
            "2-red-empty-diamonds",
            "2-red-empty-ovals",
            "2-red-empty-squiggles",
            "2-red-solid-diamonds",
            "2-red-solid-ovals",
            "2-red-solid-squiggles",
            "2-red-striped-diamonds",
            "2-red-striped-ovals",
            "2-red-striped-squiggles",
            "3-green-empty-diamonds",
            "3-green-empty-ovals",
            "3-green-empty-squiggles",
            "3-green-solid-diamonds",
            "3-green-solid-ovals",
            "3-green-solid-squiggles",
            "3-green-striped-diamonds",
            "3-green-striped-ovals",
            "3-green-striped-squiggles",
            "3-purple-empty-diamonds",
            "3-purple-empty-ovals",
            "3-purple-empty-squiggles",
            "3-purple-solid-diamonds",
            "3-purple-solid-ovals",
            "3-purple-solid-squiggles",
            "3-purple-striped-diamonds",
            "3-purple-striped-ovals",
            "3-purple-striped-squiggles",
            "3-red-empty-diamonds",
            "3-red-empty-ovals",
            "3-red-empty-squiggles",
            "3-red-solid-diamonds",
            "3-red-solid-ovals",
            "3-red-solid-squiggles",
            "3-red-striped-diamonds",
            "3-red-striped-ovals",
            "3-red-striped-squiggles")

    private val setsOfFirst12labels = setOf<String>(
            "(1-purple-empty-diamond, 1-purple-empty-oval, 1-purple-empty-squiggle)",
            "(1-green-empty-diamond, 1-green-solid-oval, 1-green-striped-squiggle)",
            "(1-green-empty-oval, 1-green-solid-oval, 1-green-striped-oval)",
            "(1-green-empty-diamond, 1-green-empty-oval, 1-green-empty-squiggle)",
            "(1-green-empty-squiggle, 1-green-solid-diamond, 1-green-striped-oval)",
            "(1-green-striped-diamond, 1-green-striped-oval, 1-green-striped-squiggle)",
            "(1-green-empty-diamond, 1-green-solid-diamond, 1-green-striped-diamond)",
            "(1-green-empty-squiggle, 1-green-solid-oval, 1-green-striped-diamond)",
            "(1-green-empty-oval, 1-green-solid-diamond, 1-green-striped-squiggle)",
            "(1-green-empty-diamond, 1-green-solid-squiggle, 1-green-striped-oval)",
            "(1-green-empty-oval, 1-green-solid-squiggle, 1-green-striped-diamond)",
            "(1-green-empty-squiggle, 1-green-solid-squiggle, 1-green-striped-squiggle)",
            "(1-green-solid-diamond, 1-green-solid-oval, 1-green-solid-squiggle)")

    @Test
    fun cardNumberApi() {
        // to String
        assertTrue(CardNumber.ONE.toString() == "1")
        assertTrue(CardNumber.TWO.toString() == "2")
        assertTrue(CardNumber.THREE.toString() == "3")
        // fromString
        assertTrue(CardNumber.fromString("1") == CardNumber.ONE)
        assertTrue(CardNumber.fromString("2") == CardNumber.TWO)
        assertTrue(CardNumber.fromString("3") == CardNumber.THREE)
        assertTrue(CardNumber.fromString("300") == null)
        // fromInt
        assertTrue(CardNumber.fromInt(CardNumber.ONE.code) == CardNumber.ONE)
        assertTrue(CardNumber.fromInt(CardNumber.TWO.code) == CardNumber.TWO)
        assertTrue(CardNumber.fromInt(CardNumber.THREE.code) == CardNumber.THREE)
        assertTrue(CardNumber.fromInt(300) == null)
        // next & previous
        for (сardNumber in listOf<CardNumber>(CardNumber.ONE, CardNumber.TWO, CardNumber.THREE)) {
            assertTrue(CardNumber.next(сardNumber) != сardNumber)
            assertTrue(CardNumber.previous(сardNumber) != сardNumber)
            assertTrue(CardNumber.next(сardNumber) != CardNumber.previous(сardNumber))
        }
    }

    @Test
    fun cardColorApi() {
        // to String
        assertTrue(CardColor.RED.toString() == "red")
        assertTrue(CardColor.GREEN.toString() == "green")
        assertTrue(CardColor.PURPLE.toString() == "purple")
        // fromString
        assertTrue(CardColor.fromString("red") == CardColor.RED)
        assertTrue(CardColor.fromString("green") == CardColor.GREEN)
        assertTrue(CardColor.fromString("purple") == CardColor.PURPLE)
        assertTrue(CardColor.fromString("blue") == null)
        // fromInt
        assertTrue(CardColor.fromInt(CardColor.RED.code) == CardColor.RED)
        assertTrue(CardColor.fromInt(CardColor.GREEN.code) == CardColor.GREEN)
        assertTrue(CardColor.fromInt(CardColor.PURPLE.code) == CardColor.PURPLE)
        assertTrue(CardColor.fromInt(300) == null)
        // next & previous
        for (cardColor in listOf<CardColor>(CardColor.RED, CardColor.GREEN, CardColor.PURPLE)) {
            assertTrue(CardColor.next(cardColor) != cardColor)
            assertTrue(CardColor.previous(cardColor) != cardColor)
            assertTrue(CardColor.next(cardColor) != CardColor.previous(cardColor))
        }
    }
    @Test
    fun cardShadingApi() {
        // to String
        assertTrue(CardShading.SOLID.toString() == "solid")
        assertTrue(CardShading.STRIPED.toString() == "striped")
        assertTrue(CardShading.EMPTY.toString() == "empty")
        // fromString
        assertTrue(CardShading.fromString("solid") == CardShading.SOLID)
        assertTrue(CardShading.fromString("striped") == CardShading.STRIPED)
        assertTrue(CardShading.fromString("empty") == CardShading.EMPTY)
        assertTrue(CardShading.fromString("other") == null)
        // fromInt
        assertTrue(CardShading.fromInt(CardShading.SOLID.code) == CardShading.SOLID)
        assertTrue(CardShading.fromInt(CardShading.STRIPED.code) == CardShading.STRIPED)
        assertTrue(CardShading.fromInt(CardShading.EMPTY.code) == CardShading.EMPTY)
        assertTrue(CardShading.fromInt(300) == null)
        // next & previous
        for (сardShading in listOf<CardShading>(CardShading.SOLID, CardShading.STRIPED, CardShading.EMPTY)) {
            assertTrue(CardShading.next(сardShading) != сardShading)
            assertTrue(CardShading.previous(сardShading) != сardShading)
            assertTrue(CardShading.next(сardShading) != CardShading.previous(сardShading))
        }
    }

    @Test
    fun cardShapeApi() {
        // to String
        assertTrue(CardShape.SQUIGGLE.toString() == "squiggle")
        assertTrue(CardShape.DIAMOND.toString() == "diamond")
        assertTrue(CardShape.OVAL.toString() == "oval")
        // fromString
        assertTrue(CardShape.fromString("squiggle") == CardShape.SQUIGGLE)
        assertTrue(CardShape.fromString("diamond") == CardShape.DIAMOND)
        assertTrue(CardShape.fromString("oval") == CardShape.OVAL)
        assertTrue(CardShape.fromString("other") == null)
        // fromInt
        assertTrue(CardShape.fromInt(CardShape.SQUIGGLE.code) == CardShape.SQUIGGLE)
        assertTrue(CardShape.fromInt(CardShape.DIAMOND.code) == CardShape.DIAMOND)
        assertTrue(CardShape.fromInt(CardShape.OVAL.code) == CardShape.OVAL)
        assertTrue(CardShape.fromInt(300) == null)
        // next & previous
        for (cardShape in listOf<CardShape>(CardShape.SQUIGGLE, CardShape.DIAMOND, CardShape.OVAL)) {
            assertTrue(CardShape.next(cardShape) != cardShape)
            assertTrue(CardShape.previous(cardShape) != cardShape)
            assertTrue(CardShape.next(cardShape) != CardShape.previous(cardShape))
        }
    }

    @Test
    fun cardValueApi() {
        assertTrue(validLabels.size == 81)
        for (label in validLabels) {
            val cardValue = CardValue.fromString(label)
            assertTrue(cardValue != null)
            assertTrue(cardValue.toString() == label)
        }
    }

    @Test
    fun SetCombinationApi() {
        fun validateCombination(input: List<String>, expected: Set<String>) {
            val cards = hashSetOf<AbstractCard>()
            for (i in input) {
                val value = CardValue.fromString(i)
                assertTrue(value != null)
                cards.add(SimpleCard(value!!))
            }
            val sets = findAllSetCombinations(cards)
            assertTrue(sets.size == expected.size)
            for (i in sets) {
                assertTrue(expected.contains(i.toString()))
            }
        }
        // corner cases
        validateCombination(listOf<String>(), setOf<String>())
        validateCombination(listOf<String>("1-green-empty-diamond"), setOf<String>())
        validateCombination(listOf<String>("1-green-empty-diamond", "1-green-empty-oval", "1-green-empty-squiggle"),
                setOf<String>("(1-green-empty-diamond, 1-green-empty-oval, 1-green-empty-squiggle)"))
        // big test
        validateCombination(validLabels.slice(0..11), setsOfFirst12labels)
    }

    @Test
    fun NonOverlappingSetCombinationApi() {
        val cards = hashSetOf<AbstractCard>()
        for (i in validLabels.slice(0..11)) {
            val value = CardValue.fromString(i)
            assertTrue(value != null)
            cards.add(SimpleCard(value!!))
        }
        val res = findAllNonOverlappingSetCombinations(findAllSetCombinations(cards))
        assertTrue(res.size > 0)
        // TBD - check that thereare no duplicated res
    }
}
