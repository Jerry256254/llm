package com.kuclab.rec

import java.io.File

data class Recording(
    val id: Long,
    val file: File,
    val name: String,
    val number: String,
    val date: Long,
    val durationSec: Long,
    val incoming: Boolean
)
