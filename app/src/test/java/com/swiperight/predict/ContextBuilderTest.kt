package com.swiperight.predict

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextBuilderTest {
    private fun b(s: String) = ContextBuilder.build(s)

    @Test
    fun emptyIsDocumentStart() {
        val c = b("")
        assertEquals("", c.context); assertEquals("", c.prefix)
        assertTrue(c.docStart); assertFalse(c.spaceBefore); assertFalse(c.suppress)
    }

    @Test
    fun afterSpaceStripsSpaceAndWantsSpacedToken() {
        val c = b("I'm going to the ")
        assertEquals("I'm going to the", c.context)
        assertEquals("", c.prefix)
        assertTrue(c.spaceBefore); assertFalse(c.insertSpace)
    }

    @Test
    fun midWordCompletesPrefix() {
        val c = b("See you tomor")
        assertEquals("See you", c.context)
        assertEquals("tomor", c.prefix)
        assertTrue(c.spaceBefore)
    }

    @Test
    fun firstWordOfDocument() {
        val c = b("Hel")
        assertEquals("", c.context); assertEquals("Hel", c.prefix)
        assertTrue(c.docStart); assertFalse(c.spaceBefore)
    }

    @Test
    fun afterNewlineAndOpenersNoSpace() {
        assertFalse(b("Hello\n").spaceBefore)
        assertEquals("Hello\n", b("Hello\n").context)
        assertFalse(b("He said (").spaceBefore)
        assertFalse(b("He said \"").spaceBefore)
        val glued = b("He said (hel")
        assertEquals("hel", glued.prefix); assertFalse(glued.spaceBefore); assertEquals("He said (", glued.context)
    }

    @Test
    fun afterSentencePunctuationInsertsSpace() {
        val c = b("This is the end.")
        assertEquals("This is the end.", c.context)
        assertTrue(c.spaceBefore); assertTrue(c.insertSpace)
    }

    @Test
    fun onlyLast32CharsAndCutWordDropped() {
        val text = "The quick brown fox jumps over the lazy dog and then "
        val c = b(text)
        assertFalse(c.docStart)
        assertTrue(c.context.length <= ContextBuilder.WINDOW_CHARS)
        // window = "ps over the lazy dog and then " -> leading fragment "ps" dropped
        assertEquals("over the lazy dog and then", c.context)
    }

    @Test
    fun windowStartingAtWordBoundaryKeepsFirstWord() {
        // 32-char window begins exactly at "jumps": nothing is a fragment.
        val window = "jumps over the lazy dog, it was "
        check(window.length == 32)
        val c = b("The quick brown fox $window")
        assertEquals("jumps over the lazy dog, it was", c.context)
    }

    @Test
    fun surrogatePairNotSplit() {
        val text = "🎉".repeat(20) + " party time "
        val c = b(text)
        assertFalse(c.context.isNotEmpty() && Character.isLowSurrogate(c.context[0]))
    }

    @Test
    fun contractionInProgressIsThePrefix() {
        val a = b("I think it'")
        assertEquals("it'", a.prefix); assertEquals("I think", a.context); assertTrue(a.spaceBefore); assertFalse(a.suppress)
        assertEquals("don't", b("I don't").prefix)
        assertEquals("you’r", b("I know you’r").prefix)
        assertEquals("hel", b("he said 'hel").prefix) // opening quote, not a contraction
    }

    @Test
    fun nonWordsSuppressed() {
        assertTrue(b("abc123de").suppress)
        assertTrue(b("x".repeat(40)).suppress)
    }

    @Test
    fun leadingSpaceOnlyAtStart() {
        val c = b(" ")
        assertTrue(c.docStart); assertFalse(c.spaceBefore); assertEquals("", c.context)
    }
}
