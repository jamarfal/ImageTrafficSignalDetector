package org.example.proyectobase;

import android.os.Bundle;
import android.preference.PreferenceActivity;

/**
 * Created by jamarfal on 19/4/17.
 */

public class Preferencias extends PreferenceActivity {
    @SuppressWarnings("deprecation")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.preferencias);
    }
}
