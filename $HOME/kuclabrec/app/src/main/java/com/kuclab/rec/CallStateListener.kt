package com.kuclab.rec

import android.content.Context
import android.content.Intent
import android.telephony.PhoneStateListener
import android.telephony.TelephonyManager
import android.util.Log

class CallStateListener(private val context: Context) : PhoneStateListener() {

    private var recording = false

    override fun onCallStateChanged(state: Int, phoneNumber: String?) {
        super.onCallStateChanged(state, phoneNumber)
        when (state) {
            TelephonyManager.CALL_STATE_OFFHOOK -> {
                if (!recording && RecordingService.isRunning) {
                    startRec(phoneNumber, true)
                    recording = true
                }
            }
            TelephonyManager.CALL_STATE_IDLE -> {
                if (recording) {
                    stopRec()
                    recording = false
                }
            }
            TelephonyManager.CALL_STATE_RINGING -> {
                // příchozí číslo uložíme pro případ OFFHOOK
                ringingNumber = phoneNumber
            }
        }
    }

    private fun startRec(number: String?, incoming: Boolean) {
        val num = number ?: ringingNumber ?: "unknown"
        val intent = Intent(context, RecordingService::class.java).apply {
            action = RecordingService.ACTION_START
            putExtra(RecordingService.EXTRA_NUMBER, num)
            putExtra(RecordingService.EXTRA_INCOMING, incoming)
        }
        context.startForegroundService(intent)
    }

    private fun stopRec() {
        val intent = Intent(context, RecordingService::class.java).apply {
            action = RecordingService.ACTION_STOP
        }
        context.startService(intent)
    }

    companion object {
        var ringingNumber: String? = null
    }
}
