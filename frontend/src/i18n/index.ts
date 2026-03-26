import { createI18n } from 'vue-i18n'
import en from './locales/en/messages.json'
import es from './locales/es/messages.json'

export const i18n = createI18n({
  legacy: false,
  locale: 'en',
  fallbackLocale: 'en',
  messages: { en, es },
})
