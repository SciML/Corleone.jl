<template>
  <div class="gallery-container">
    <div class="controls">
      <input 
        v-model="searchQuery" 
        type="text" 
        placeholder="Search tutorials..." 
        class="search-input"
      />
      
      <div class="filter-bar">
        <button 
          class="filter-chip" 
          :class="{ active: selectedTag === null }"
          @click="selectedTag = null"
        >
          All
        </button>
        <button 
          v-for="tag in allTags" 
          :key="tag" 
          class="filter-chip"
          :class="{ active: selectedTag === tag }"
          @click="selectedTag = (selectedTag === tag ? null : tag)"
        >
          {{ tag }}
        </button>
      </div>
    </div>

    <div class="grid">
      <div v-for="item in filteredData" :key="item.link" class="card">
        <div class="card-header">
          <span class="icon">{{ item.icon || '📄' }}</span>
          <h3>{{ item.title }}</h3>
        </div>

        <div class="tag-list">
          <span v-for="tag in item.tags" :key="tag" class="tag-pill">
            {{ tag }}
          </span>
        </div>

        <p class="description">{{ item.description }}</p>
        
        <div class="card-footer">
          <a :href="item.link" class="link">Read Tutorial →</a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface Tutorial {
  title: string;
  description: string;
  link: string;
  icon?: string;
  tags?: string[];
}

const props = defineProps<{ items: Tutorial[] }>()

const searchQuery = ref('')
const selectedTag = ref<string | null>(null)

// Automatically extract unique tags from your Julia data
const allTags = computed(() => {
  const tags = new Set<string>()
  props.items.forEach(item => {
    item.tags?.forEach(t => tags.add(t))
  })
  return Array.from(tags).sort()
})

const filteredData = computed(() => {
  return props.items.filter(item => {
    const matchesSearch = item.title.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
                          item.description.toLowerCase().includes(searchQuery.value.toLowerCase())
    const matchesTag = !selectedTag.value || item.tags?.includes(selectedTag.value)
    
    return matchesSearch && matchesTag
  })
})
</script>

<style scoped>
.controls {
  margin-bottom: 2rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.search-input {
  width: 100%;
  padding: 12px 16px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-alt);
}

/* Filter Chips Styling */
.filter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.filter-chip {
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 500;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  transition: all 0.2s ease;
  cursor: pointer;
}

.filter-chip:hover {
  border-color: var(--vp-c-brand-1);
}

.filter-chip.active {
  background: var(--vp-c-brand-1);
  color: white;
  border-color: var(--vp-c-brand-1);
}

/* Card Styling */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

.card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
}

.tag-pill {
  font-size: 0.7rem;
  background: var(--vp-c-default-soft);
  padding: 2px 8px;
  border-radius: 4px;
  margin-right: 5px;
  color: var(--vp-c-text-2);
}

.description {
  margin: 1rem 0;
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  flex-grow: 1;
}

.link {
  font-weight: bold;
  color: var(--vp-c-brand-1);
  text-decoration: none;
}
</style>