package com.kuclab.rec

import android.content.Context
import android.provider.ContactsContract
import java.io.File

object RecorderStorage {

    fun recordingsDir(context: Context): File {
        val dir = File(context.getExternalFilesDir(null), "recordings")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    fun newFile(context: Context, number: String, incoming: Boolean): File {
        val ts = System.currentTimeMillis()
        val safe = number.replace("[^0-9+]".toRegex(), "").ifEmpty { "unknown" }
        val dir = recordingsDir(context)
        return File(dir, "${if (incoming) "in" else "out"}_${safe}_$ts.mp3")
    }

    fun listRecordings(context: Context): List<Recording> {
        val dir = recordingsDir(context)
        val files = dir.listFiles { f -> f.extension == "mp3" } ?: return emptyList()
        return files.sortedByDescending { it.lastModified() }.mapIndexed { idx, f ->
            val (incoming, number) = parseName(f.name)
            val name = lookupContact(context, number)
            Recording(
                id = idx.toLong(),
                file = f,
                name = name ?: context.getString(R.string.name),
                number = number,
                date = f.lastModified(),
                durationSec = 0L,
                incoming = incoming
            )
        }
    }

    private fun parseName(fileName: String): Pair<Boolean, String> {
        val parts = fileName.removeSuffix(".mp3").split("_")
        val incoming = parts.firstOrNull() == "in"
        val number = parts.getOrNull(1) ?: "unknown"
        return incoming to number
    }

    private fun lookupContact(context: Context, number: String): String? {
        if (number == "unknown") return null
        val uri = ContactsContract.CommonDataKinds.Phone.CONTENT_URI
        val projection = arrayOf(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
        val sel = "${ContactsContract.CommonDataKinds.Phone.NUMBER} = ?"
        context.contentResolver.query(uri, projection, sel, arrayOf(number), null)?.use { c ->
            if (c.moveToFirst()) return c.getString(0)
        }
        return null
    }
}
