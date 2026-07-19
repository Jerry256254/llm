package com.kuclab.rec

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class OutgoingCallReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
        if (intent?.action == Intent.ACTION_NEW_OUTGOING_CALL) {
            val number = intent.getStringExtra(Intent.EXTRA_PHONE_NUMBER)
            if (context != null && RecordingService.isRunning) {
                CallStateListener.ringingNumber = number
                val i = Intent(context, RecordingService::class.java).apply {
                    action = RecordingService.ACTION_START
                    putExtra(RecordingService.EXTRA_NUMBER, number ?: "unknown")
                    putExtra(RecordingService.EXTRA_INCOMING, false)
                }
                context.startForegroundService(i)
            }
        }
    }
}
